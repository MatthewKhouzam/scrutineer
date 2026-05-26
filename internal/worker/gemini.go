package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/genai"
)

// GeminiRunner implements SkillRunner by calling the Gemini API directly
// via the go-genai SDK with function calling in a loop.
type GeminiRunner struct {
	APIKey    string // GEMINI_API_KEY
	Model     string // fallback model; SkillJob.Model wins
	FullClone bool
	MaxTurns  int
}

var geminiTools = &genai.Tool{
	FunctionDeclarations: []*genai.FunctionDeclaration{
		{
			Name:        "read_file",
			Description: "Read the contents of a file relative to the workspace root.",
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string", "description": "Relative file path"}},
				"required":   []string{"path"},
			},
		},
		{
			Name:        "write_file",
			Description: "Write content to a file relative to the workspace root. Creates parent directories.",
			ParametersJsonSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Relative file path"},
					"content": map[string]any{"type": "string", "description": "File content"},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			Name:        "list_directory",
			Description: "List files and directories at a path relative to the workspace root.",
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string", "description": "Relative directory path (use . for root)"}},
				"required":   []string{"path"},
			},
		},
		{
			Name:        "run_command",
			Description: "Run a shell command in the workspace root. Returns stdout+stderr.",
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"command": map[string]any{"type": "string", "description": "Shell command to execute"}},
				"required":   []string{"command"},
			},
		},
		{
			Name:        "web_fetch",
			Description: "Fetch a URL and return the response body as text.",
			ParametersJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"url": map[string]any{"type": "string", "description": "URL to fetch"}},
				"required":   []string{"url"},
			},
		},
	},
}

func (g GeminiRunner) RunSkill(ctx context.Context, sj SkillJob, emit func(Event)) (SkillResult, error) {
	src, err := ensureClone(ctx, sj.Repo, sj.WorkRoot, g.FullClone, sj.Ref, emit)
	if err != nil {
		return SkillResult{}, err
	}
	commit := gitHead(src)
	work := sj.WorkRoot

	var outPath string
	if sj.OutputFile != "" {
		outPath = filepath.Join(work, sj.OutputFile)
		_ = os.Remove(outPath)
	}

	// Build system prompt from the staged skill.
	skillMD, err := os.ReadFile(filepath.Join(sj.SkillDir, "SKILL.md"))
	if err != nil {
		return SkillResult{}, fmt.Errorf("read skill: %w", err)
	}
	schemaTxt := ""
	if b, err2 := os.ReadFile(filepath.Join(sj.SkillDir, "schema.json")); err2 == nil {
		schemaTxt = "\n\n## Output Schema\n```json\n" + string(b) + "\n```"
	}

	systemPrompt := fmt.Sprintf(
		"You are an automated analysis agent. Execute the following skill on the repository cloned at ./src.\n\n"+
			"--- SKILL ---\n%s\n--- END SKILL ---%s\n\n"+
			"The workspace root is your working directory. The cloned repository is at ./src/. "+
			"Context about the repository is in ./context.json. "+
			"Write your output to ./%s as specified by the skill.",
		string(skillMD), schemaTxt, sj.OutputFile,
	)

	model := sj.Model
	if model == "" {
		model = g.Model
	}
	if model == "" {
		model = "gemini-2.5-flash"
	}

	apiKey := g.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return SkillResult{Commit: commit}, fmt.Errorf("gemini client: %w", err)
	}

	emit(Event{Kind: KindText, Text: fmt.Sprintf("$ gemini [%s] <skill:%s>", model, sj.Name)})

	contents := []*genai.Content{
		genai.NewContentFromText(buildSkillPrompt(sj.Name, sj.OutputFile), "user"),
	}

	config := &genai.GenerateContentConfig{
		SystemInstruction: genai.NewContentFromText(systemPrompt, "user"),
		Tools:             []*genai.Tool{geminiTools},
	}

	maxTurns := effectiveMaxTurns(sj.MaxTurns, g.MaxTurns)
	var totalInput, totalOutput int

	for turn := 0; turn < maxTurns; turn++ {
		resp, err := client.Models.GenerateContent(ctx, model, contents, config)
		if err != nil {
			return SkillResult{Commit: commit}, fmt.Errorf("gemini generate: %w", err)
		}

		if resp.UsageMetadata != nil {
			totalInput += int(resp.UsageMetadata.PromptTokenCount)
			totalOutput += int(resp.UsageMetadata.CandidatesTokenCount)
		}

		if len(resp.Candidates) == 0 {
			return SkillResult{Commit: commit}, fmt.Errorf("gemini: empty response")
		}

		candidate := resp.Candidates[0]
		if candidate.Content == nil {
			break
		}

		// Append the model's response to the conversation.
		contents = append(contents, candidate.Content)

		// Collect function calls and text from parts.
		var functionCalls []*genai.FunctionCall
		for _, part := range candidate.Content.Parts {
			if part.Text != "" && !part.Thought {
				emit(Event{Kind: KindText, Text: part.Text})
			}
			if part.Thought && part.Text != "" {
				emit(Event{Kind: KindThinking, Text: part.Text})
			}
			if part.FunctionCall != nil {
				functionCalls = append(functionCalls, part.FunctionCall)
			}
		}

		// No function calls means the model is done.
		if len(functionCalls) == 0 {
			break
		}

		// Execute function calls and build response parts.
		var responseParts []*genai.Part
		for _, fc := range functionCalls {
			argsJSON, _ := json.Marshal(fc.Args)
			emit(Event{Kind: KindTool, Tool: fc.Name, Text: truncate(string(argsJSON))})
			result := g.executeTool(ctx, work, fc.Name, fc.Args)
			responseParts = append(responseParts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       fc.ID,
					Name:     fc.Name,
					Response: map[string]any{"output": result},
				},
			})
		}
		contents = append(contents, &genai.Content{
			Role:  "user",
			Parts: responseParts,
		})

		if candidate.FinishReason == genai.FinishReasonStop {
			break
		}
	}

	emit(Event{
		Kind:  KindResult,
		Text:  "done",
		Usage: Usage{InputTokens: totalInput, OutputTokens: totalOutput},
	})

	res := SkillResult{Commit: commit}
	if outPath != "" {
		res.Report = readCappedReport(outPath, emit)
	}
	return res, nil
}

func (g GeminiRunner) executeTool(ctx context.Context, workRoot, name string, args map[string]any) string {
	str := func(key string) string {
		if v, ok := args[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
		return ""
	}

	switch name {
	case "read_file":
		return g.toolReadFile(workRoot, str("path"))
	case "write_file":
		return g.toolWriteFile(workRoot, str("path"), str("content"))
	case "list_directory":
		return g.toolListDir(workRoot, str("path"))
	case "run_command":
		return g.toolRunCommand(ctx, workRoot, str("command"))
	case "web_fetch":
		return g.toolWebFetch(ctx, str("url"))
	default:
		return fmt.Sprintf("unknown tool: %s", name)
	}
}

func (g GeminiRunner) toolReadFile(workRoot, path string) string {
	full := filepath.Join(workRoot, filepath.Clean(path))
	if full != workRoot && !strings.HasPrefix(full, workRoot+string(os.PathSeparator)) {
		return "error: path escapes workspace"
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	if len(b) > maxReportBytes {
		return string(b[:maxReportBytes]) + "\n[truncated]"
	}
	return string(b)
}

func (g GeminiRunner) toolWriteFile(workRoot, path, content string) string {
	full := filepath.Join(workRoot, filepath.Clean(path))
	if full != workRoot && !strings.HasPrefix(full, workRoot+string(os.PathSeparator)) {
		return "error: path escapes workspace"
	}
	if err := os.MkdirAll(filepath.Dir(full), dirPerm); err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	if err := os.WriteFile(full, []byte(content), filePerm); err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return "ok"
}

func (g GeminiRunner) toolListDir(workRoot, path string) string {
	full := filepath.Join(workRoot, filepath.Clean(path))
	if full != workRoot && !strings.HasPrefix(full, workRoot+string(os.PathSeparator)) {
		return "error: path escapes workspace"
	}
	entries, err := os.ReadDir(full)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	var sb strings.Builder
	for _, e := range entries {
		if e.IsDir() {
			sb.WriteString(e.Name() + "/\n")
		} else {
			sb.WriteString(e.Name() + "\n")
		}
	}
	return sb.String()
}

func (g GeminiRunner) toolRunCommand(ctx context.Context, workRoot, command string) string {
	if command == "" {
		return "error: empty command"
	}
	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = workRoot
	out, err := cmd.CombinedOutput()
	result := string(out)
	if err != nil {
		result += "\nexit: " + err.Error()
	}
	if len(result) > maxReportBytes {
		result = result[:maxReportBytes] + "\n[truncated]"
	}
	return result
}

func (g GeminiRunner) toolWebFetch(ctx context.Context, url string) string {
	if url == "" {
		return "error: empty url"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxReportBytes)))
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return string(b)
}
