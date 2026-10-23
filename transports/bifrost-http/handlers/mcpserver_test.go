package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fasthttp/router"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	coremcp "github.com/maximhq/bifrost/core/mcp"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

func TestConvertToolFunctionParametersToMCPInputSchemaPreservesDefs(t *testing.T) {
	params := &schemas.ToolFunctionParameters{
		Type: "object",
		Properties: schemas.NewOrderedMapFromPairs(
			schemas.KV("preferences", map[string]any{"$ref": "#/$defs/Preferences"}),
		),
		Required: []string{"preferences"},
		Defs: schemas.NewOrderedMapFromPairs(
			schemas.KV("Preferences", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"startHour": map[string]any{"type": "string"},
				},
			}),
		),
	}

	inputSchema := convertToolFunctionParametersToMCPInputSchema(params)

	require.Contains(t, inputSchema.Defs, "Preferences")
	data, err := json.Marshal(inputSchema)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"$defs"`)
	assert.Contains(t, string(data), `"$ref":"#/$defs/Preferences"`)
}

func TestConvertToolFunctionParametersToMCPInputSchemaPreservesLegacyDefinitionsAsDefs(t *testing.T) {
	params := &schemas.ToolFunctionParameters{
		Type: "object",
		Properties: schemas.NewOrderedMapFromPairs(
			schemas.KV("preferences", map[string]any{"$ref": "#/$defs/Preferences"}),
		),
		Definitions: schemas.NewOrderedMapFromPairs(
			schemas.KV("Preferences", map[string]any{"type": "object"}),
		),
	}

	inputSchema := convertToolFunctionParametersToMCPInputSchema(params)

	require.Contains(t, inputSchema.Defs, "Preferences")
	data, err := json.Marshal(inputSchema)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"$defs"`)
}

// The MCP spec uses isError to tell the model a tool call failed. The gateway
// executes the upstream tool and then rebuilds the reply for its own client, so
// the flag has to survive that rebuild: without it a failed call reads as a
// success, and anything counting tool errors undercounts them.
func TestToolCallResultFromMessagePreservesUpstreamIsError(t *testing.T) {
	failure := "{\"error\": \"bad input\"}"
	msg := &schemas.ChatMessage{
		Content:         &schemas.ChatMessageContent{ContentStr: &failure},
		ChatToolMessage: &schemas.ChatToolMessage{IsError: schemas.Ptr(true)},
	}

	result := toolCallResultFromMessage(msg)

	require.NotNil(t, result)
	assert.True(t, result.IsError, "upstream isError must reach the gateway's client")
	require.Len(t, result.Content, 1)
	text, ok := result.Content[0].(mcp.TextContent)
	require.True(t, ok, "content should stay text")
	assert.Equal(t, failure, text.Text, "the error text must not be lost with the flag")
}

// A tool that succeeded must not be reported as failed, so the flag is only
// raised when the upstream server raised it.
func TestToolCallResultFromMessageLeavesSuccessAlone(t *testing.T) {
	body := "ok"
	for name, toolMsg := range map[string]*schemas.ChatToolMessage{
		"absent tool message": nil,
		"flag unset":          {},
		"flag false":          {IsError: schemas.Ptr(false)},
	} {
		t.Run(name, func(t *testing.T) {
			result := toolCallResultFromMessage(&schemas.ChatMessage{
				Content:         &schemas.ChatMessageContent{ContentStr: &body},
				ChatToolMessage: toolMsg,
			})

			require.NotNil(t, result)
			assert.False(t, result.IsError)
			require.Len(t, result.Content, 1)
			text, ok := result.Content[0].(mcp.TextContent)
			require.True(t, ok)
			assert.Equal(t, body, text.Text)
		})
	}
}

// Structured replies arrive as content blocks rather than a single string, and
// the flag rides on the message either way.
func TestToolCallResultFromMessageJoinsContentBlocksAndKeepsIsError(t *testing.T) {
	first, second := "partial ", "failure"
	msg := &schemas.ChatMessage{
		Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
			{Type: schemas.ChatContentBlockTypeText, Text: &first},
			{Type: schemas.ChatContentBlockTypeText, Text: &second},
		}},
		ChatToolMessage: &schemas.ChatToolMessage{IsError: schemas.Ptr(true)},
	}

	result := toolCallResultFromMessage(msg)

	require.NotNil(t, result)
	assert.True(t, result.IsError)
	require.Len(t, result.Content, 1)
	text, ok := result.Content[0].(mcp.TextContent)
	require.True(t, ok)
	assert.Equal(t, "partial failure", text.Text)
}

// A nil message is the executor returning nothing; the gateway still owes its
// client a well-formed result rather than a panic.
func TestToolCallResultFromMessageHandlesNilMessage(t *testing.T) {
	result := toolCallResultFromMessage(nil)

	require.NotNil(t, result)
	assert.False(t, result.IsError)
}

// chainToolManager only adapts the transport interface to the real MCP manager.
// Discovery, permissions, upstream HTTP calls and result conversion stay on the
// production path at both hops; no tool result is manufactured by this adapter.
type chainToolManager struct{ manager *coremcp.MCPManager }

func (m *chainToolManager) GetAvailableMCPTools(ctx context.Context) []schemas.ChatTool {
	return m.manager.GetAvailableTools(schemas.NewBifrostContext(ctx, schemas.NoDeadline))
}

func (m *chainToolManager) ExecuteChatMCPTool(ctx context.Context, call *schemas.ChatAssistantMessageToolCall) (*schemas.ChatMessage, *schemas.BifrostError) {
	return m.manager.ExecuteChatTool(schemas.NewBifrostContext(ctx, schemas.NoDeadline), call)
}

func (m *chainToolManager) ExecuteResponsesMCPTool(ctx context.Context, call *schemas.ResponsesToolMessage) (*schemas.ResponsesMessage, *schemas.BifrostError) {
	return m.manager.ExecuteResponsesTool(schemas.NewBifrostContext(ctx, schemas.NoDeadline), call)
}

// startChainGateway serves the actual Bifrost routes on an ephemeral TCP port.
func startChainGateway(t *testing.T, ctx context.Context, upstream *schemas.MCPClientConfig) (*coremcp.MCPManager, string) {
	t.Helper()
	manager := coremcp.NewMCPManager(ctx, schemas.MCPConfig{ToolSyncInterval: time.Hour}, nil, &mockLogger{}, nil)
	t.Cleanup(func() { require.NoError(t, manager.Cleanup()) })
	require.NoError(t, manager.AddClient(ctx, upstream))
	handler, err := NewMCPServerHandler(ctx, &lib.Config{ClientConfig: &configstore.ClientConfig{}}, &chainToolManager{manager}, nil, nil, nil)
	require.NoError(t, err)
	r := router.New()
	handler.RegisterRoutes(r)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	httpServer := &fasthttp.Server{Handler: r.Handler}
	done := make(chan error, 1)
	go func() { done <- httpServer.Serve(listener) }()
	t.Cleanup(func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		assert.NoError(t, httpServer.ShutdownWithContext(shutdownCtx))
		select {
		case err := <-done:
			assert.NoError(t, err)
		case <-shutdownCtx.Done():
			t.Error("gateway did not stop")
		}
	})
	return manager, "http://" + listener.Addr().String() + "/mcp"
}

func chainClientConfig(name, url string) *schemas.MCPClientConfig {
	return &schemas.MCPClientConfig{
		ID: name, Name: name,
		ConnectionType:   schemas.MCPConnectionTypeHTTP,
		ConnectionString: schemas.NewSecretVar(url),
		AuthType:         schemas.MCPAuthTypeNone,
		ToolsToExecute:   []string{"*"},
		ToolSyncInterval: time.Hour,
	}
}

// chainRPC checks the HTTP and JSON-RPC envelopes separately from tool IsError.
func chainRPC(t *testing.T, ctx context.Context, client *http.Client, url, method string, params any) json.RawMessage {
	t.Helper()
	id := schemas.Ptr(1)
	if method == "notifications/initialized" {
		id = nil
	}
	body, err := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		ID      *int   `json:"id,omitempty"`
		Method  string `json:"method"`
		Params  any    `json:"params"`
	}{"2.0", id, method, params})
	require.NoError(t, err)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	if id == nil {
		require.Equal(t, http.StatusAccepted, resp.StatusCode)
		return nil
	}
	require.Equal(t, http.StatusOK, resp.StatusCode)
	var envelope struct {
		ID     int             `json:"id"`
		Result json.RawMessage `json:"result"`
		Error  json.RawMessage `json:"error"`
	}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&envelope))
	require.Equal(t, 1, envelope.ID)
	require.Empty(t, envelope.Error, "JSON-RPC error: %s", envelope.Error)
	require.NotEmpty(t, envelope.Result)
	return envelope.Result
}

func TestMCPGatewayChainPreservesToolErrors(t *testing.T) {
	SetLogger(&mockLogger{})
	for _, tc := range []struct {
		name         string
		text         string
		leafError    bool
		denyAtShared bool
	}{
		{name: "upstream tool error", text: "bad input", leafError: true},
		{name: "success", text: "ok"},
		{name: "success containing error text", text: "error count: 0"},
		{name: "shared execution refusal", denyAtShared: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			t.Cleanup(cancel)
			var calls atomic.Int32
			leaf := server.NewMCPServer("chain-leaf", "1", server.WithToolCapabilities(false))
			leaf.AddTool(mcp.NewTool("probe", mcp.WithString("message", mcp.Required())),
				func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
					calls.Add(1)
					text, _ := req.GetArguments()["message"].(string)
					result := mcp.NewToolResultText(text)
					result.IsError = tc.leafError
					return result, nil
				})
			leafHTTP := httptest.NewServer(server.NewStreamableHTTPServer(leaf))
			t.Cleanup(leafHTTP.Close)
			leafConfig := chainClientConfig("leaf", leafHTTP.URL)
			shared, sharedURL := startChainGateway(t, ctx, leafConfig)
			_, localURL := startChainGateway(t, ctx, chainClientConfig("estate", sharedURL))
			transport := &http.Transport{}
			t.Cleanup(transport.CloseIdleConnections)
			client := &http.Client{Transport: transport, Timeout: 5 * time.Second}
			chainRPC(t, ctx, client, localURL, "initialize", mcp.InitializeParams{
				ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
				Capabilities:    mcp.ClientCapabilities{},
				ClientInfo:      mcp.Implementation{Name: "chain-test", Version: "1"},
			})
			chainRPC(t, ctx, client, localURL, "notifications/initialized", struct{}{})
			var listed mcp.ListToolsResult
			require.NoError(t, json.Unmarshal(chainRPC(t, ctx, client, localURL, "tools/list", struct{}{}), &listed))
			require.Len(t, listed.Tools, 1)
			require.Equal(t, "estate-leaf-probe", listed.Tools[0].Name)
			require.Zero(t, calls.Load(), "discovery must not execute the tool")
			if tc.denyAtShared {
				// Keep both handler registries and the local discovery cache intact;
				// the real shared execution gate must reject the stale callable name.
				updated := *leafConfig
				updated.ToolsToExecute = []string{}
				require.NoError(t, shared.UpdateClient(leafConfig.ID, &updated))
			}
			resultJSON := chainRPC(t, ctx, client, localURL, "tools/call", mcp.CallToolParams{
				Name:      listed.Tools[0].Name,
				Arguments: map[string]any{"message": tc.text},
			})
			var result struct {
				IsError bool              `json:"isError"`
				Content []mcp.TextContent `json:"content"`
			}
			require.NoError(t, json.Unmarshal(resultJSON, &result))
			assert.Equal(t, tc.leafError || tc.denyAtShared, result.IsError, "error flag must survive both gateways: %s", resultJSON)
			require.Len(t, result.Content, 1)
			assert.Equal(t, "text", result.Content[0].Type)
			if tc.denyAtShared {
				assert.Contains(t, result.Content[0].Text, "not permitted")
				assert.Zero(t, calls.Load(), "shared refusal must not reach the leaf")
			} else {
				assert.Equal(t, tc.text, result.Content[0].Text)
				assert.EqualValues(t, 1, calls.Load(), "one logical call must invoke the leaf once")
			}
		})
	}
}
