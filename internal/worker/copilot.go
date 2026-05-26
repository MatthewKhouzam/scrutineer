package worker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	copilot "github.com/github/copilot-sdk/go"
)

// CopilotRunner implements SkillRunner by spawning the Copilot CLI via the
// copilot-sdk-go library. The CLI handles its own tools (file read/write,
// shell commands, etc.) so we just send the prompt and collect events.
type CopilotRunner struct {
	FullClone bool
	MaxTurns  int
}

func (c CopilotRunner) RunSkill(ctx context.Context, sj SkillJob, emit func(Event)) (SkillResult, error) {
	src, err := ensureClone(ctx, sj.Repo, sj.WorkRoot, c.FullClone, sj.Ref, emit)
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

	absWork, _ := filepath.Abs(work)

	model := sj.Model
	if model == "" {
		model = "gpt-4.1"
	}

	emit(Event{Kind: KindText, Text: fmt.Sprintf("$ copilot [%s] <skill:%s>", model, sj.Name)})

	client := copilot.NewClient(&copilot.ClientOptions{
		Cwd: absWork,
	})
	if err := client.Start(ctx); err != nil {
		return SkillResult{Commit: commit}, fmt.Errorf("copilot start: %w", err)
	}
	defer client.Stop() //nolint:errcheck

	session, err := client.CreateSession(ctx, &copilot.SessionConfig{
		Model:               model,
		WorkingDirectory:    absWork,
		OnPermissionRequest: copilot.PermissionHandler.ApproveAll,
	})
	if err != nil {
		return SkillResult{Commit: commit}, fmt.Errorf("copilot create session: %w", err)
	}
	defer session.Disconnect() //nolint:errcheck

	// Subscribe to events for streaming output.
	var (
		mu          sync.Mutex
		totalInput  int
		totalOutput int
		turns       int
		idleCh      = make(chan struct{}, 1)
		errCh       = make(chan error, 1)
	)

	session.On(func(event copilot.SessionEvent) {
		switch d := event.Data.(type) {
		case *copilot.AssistantMessageData:
			if d.Content != "" {
				emit(Event{Kind: KindText, Text: d.Content})
			}
			if d.ReasoningText != nil && *d.ReasoningText != "" {
				emit(Event{Kind: KindThinking, Text: *d.ReasoningText})
			}
		case *copilot.ToolExecutionStartData:
			emit(Event{Kind: KindTool, Tool: d.ToolName, Text: fmt.Sprintf("%v", d.Arguments)})
		case *copilot.AssistantUsageData:
			mu.Lock()
			if d.InputTokens != nil {
				totalInput += int(*d.InputTokens)
			}
			if d.OutputTokens != nil {
				totalOutput += int(*d.OutputTokens)
			}
			turns++
			mu.Unlock()
		case *copilot.SessionIdleData:
			select {
			case idleCh <- struct{}{}:
			default:
			}
		case *copilot.SessionErrorData:
			select {
			case errCh <- fmt.Errorf("copilot session error: %s", d.Message):
			default:
			}
		}
	})

	prompt := buildSkillPrompt(sj.Name, sj.OutputFile)
	if _, err := session.Send(ctx, copilot.MessageOptions{Prompt: prompt}); err != nil {
		return SkillResult{Commit: commit}, fmt.Errorf("copilot send: %w", err)
	}

	// Wait for idle or error.
	select {
	case <-ctx.Done():
		return SkillResult{Commit: commit}, ctx.Err()
	case err := <-errCh:
		return SkillResult{Commit: commit}, err
	case <-idleCh:
	}

	mu.Lock()
	usage := Usage{InputTokens: totalInput, OutputTokens: totalOutput}
	t := turns
	mu.Unlock()

	emit(Event{Kind: KindResult, Text: "done", Turns: t, Usage: usage})

	res := SkillResult{Commit: commit}
	if outPath != "" {
		res.Report = readCappedReport(outPath, emit)
	}
	return res, nil
}
