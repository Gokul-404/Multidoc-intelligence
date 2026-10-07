import React, { useState, useRef, useEffect } from "react";
import { ChatMessage, Citation } from "../types";
import {
  Send,
  Sparkles,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  ExternalLink,
  ChevronRight,
  Terminal,
  FileText,
  Clock,
  CheckCircle2,
  XCircle,
  HelpCircle,
} from "lucide-react";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => Promise<void>;
  isLoading: boolean;
  onOpenDeveloperDrawer: (metrics: any) => void;
  selectedDocCount: number;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  onOpenDeveloperDrawer,
  selectedDocCount,
}) => {
  const [inputText, setInputText] = useState("");
  const [activeCitationSnippet, setActiveCitationSnippet] = useState<Citation | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    const query = inputText.trim();
    setInputText("");
    await onSendMessage(query);
  };

  const sampleQueries = [
    "What were Alphabet's consolidated revenues in Q3 2024 and YoY growth?",
    "Under Section 8.2 of the agreement, what is the limitation of liability cap?",
    "What is the secret recipe of Coca-Cola in the document? (Tests Refusal)",
  ];

  return (
    <div className="flex-1 flex flex-col h-full bg-[#080c14] overflow-hidden relative">
      {/* Top bar info */}
      <div className="flex items-center justify-between px-6 py-2.5 border-b border-slate-800/80 bg-[#0c121e]/40 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>Multi-Hop Reasoning & Audit Pipeline</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-500 font-mono">
            {selectedDocCount === 0
              ? "Target: All documents in workspace"
              : `Target: ${selectedDocCount} specific document(s)`}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-500 font-mono">
          <span>Max Retries: 2</span>
          <span>•</span>
          <span>Strict Citation Audit Active</span>
        </div>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center space-y-5 my-auto">
            <div className="w-14 h-14 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 shadow-glow">
              <Sparkles className="w-7 h-7" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white tracking-tight">
                SentinelRAG Evidence Intelligence
              </h3>
              <p className="text-xs text-slate-400 mt-1.5 max-w-md mx-auto leading-relaxed">
                Ask deep factual questions across your uploaded research papers,
                financial filings, contracts, or technical documentation. Every answer is
                verified against retrieved evidence.
              </p>
            </div>

            {/* Quick Test Queries */}
            <div className="w-full space-y-2 pt-2">
              <p className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
                Sample Test Scenarios
              </p>
              {sampleQueries.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(q)}
                  className="w-full text-left p-3 rounded-xl bg-slate-900/60 hover:bg-slate-800/90 border border-slate-800 hover:border-blue-500/40 text-xs text-slate-300 transition-all flex items-center justify-between group"
                >
                  <span className="truncate pr-2">{q}</span>
                  <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-blue-400 shrink-0 transition-transform group-hover:translate-x-0.5" />
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.role === "user" ? "items-end" : "items-start"
              }`}
            >
              {/* Message Bubble */}
              <div
                className={`max-w-3xl rounded-2xl p-5 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white shadow-glow"
                    : "glass-panel-elevated text-slate-200 border-slate-800/90"
                }`}
              >
                {/* Header for Assistant Messages */}
                {msg.role === "assistant" && (
                  <div className="flex items-center justify-between mb-3.5 pb-2.5 border-b border-slate-800/80 text-xs">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-blue-400" />
                      <span className="font-semibold text-white">SentinelRAG Agent</span>
                      {msg.refused ? (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1">
                          <ShieldAlert className="w-3 h-3" /> Refused (Unsafe/Ungrounded)
                        </span>
                      ) : msg.grounded ? (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> Grounded & Verified
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                          Partial Evidence
                        </span>
                      )}
                    </div>

                    {/* Confidence Score Pill */}
                    {msg.confidence_score !== undefined && (
                      <div className="flex items-center gap-2 font-mono">
                        <span className="text-[11px] text-slate-400">Confidence:</span>
                        <span
                          className={`font-bold px-2 py-0.5 rounded text-xs ${
                            msg.confidence_score >= 85
                              ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                              : msg.confidence_score >= 70
                              ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                              : "bg-rose-500/15 text-rose-300 border border-rose-500/30"
                          }`}
                        >
                          {msg.confidence_score.toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Content */}
                <div className="text-sm leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </div>

                {/* Refusal Explanations Banner */}
                {msg.refused && msg.refusal_reason && (
                  <div className="mt-4 p-3 rounded-xl bg-rose-950/20 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
                    <div>
                      <span className="font-semibold block mb-0.5">Hallucination Guard Triggered</span>
                      <p className="text-slate-300 text-[11px] leading-relaxed">
                        {msg.refusal_reason}
                      </p>
                    </div>
                  </div>
                )}

                {/* Citations Grid */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-3.5 border-t border-slate-800/80">
                    <p className="text-[11px] font-mono text-slate-400 mb-2 flex items-center gap-1.5">
                      <FileText className="w-3.5 h-3.5 text-blue-400" />
                      Audited Evidence Citations ({msg.citations.length}):
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {msg.citations.map((c) => (
                        <button
                          key={c.citation_index}
                          onClick={() =>
                            setActiveCitationSnippet(
                              activeCitationSnippet?.chunk_id === c.chunk_id ? null : c
                            )
                          }
                          className={`text-left p-2.5 rounded-lg border text-xs transition-all flex items-start justify-between gap-2 ${
                            activeCitationSnippet?.chunk_id === c.chunk_id
                              ? "bg-blue-600/20 border-blue-500 text-blue-200"
                              : "bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-300"
                          }`}
                        >
                          <div className="overflow-hidden">
                            <span className="font-semibold block truncate">
                              [{c.citation_index}] {c.document_name}
                            </span>
                            <span className="text-[10px] font-mono text-slate-400">
                              Page {c.page_number} • Chunk #{c.chunk_id.slice(-6)}
                            </span>
                          </div>
                          <ChevronRight className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-1" />
                        </button>
                      ))}
                    </div>

                    {/* Expanded Citation Preview Box */}
                    {activeCitationSnippet && (
                      <div className="mt-3 p-3.5 rounded-xl bg-slate-950 border border-blue-500/40 text-xs">
                        <div className="flex items-center justify-between text-blue-400 font-mono text-[11px] mb-1.5">
                          <span>
                            Evidence Snippet from {activeCitationSnippet.document_name} (Page{" "}
                            {activeCitationSnippet.page_number})
                          </span>
                          <button
                            onClick={() => setActiveCitationSnippet(null)}
                            className="text-slate-500 hover:text-slate-300"
                          >
                            Close
                          </button>
                        </div>
                        <p className="text-slate-200 font-serif italic text-xs leading-relaxed bg-slate-900/60 p-2.5 rounded border border-slate-800">
                          "{activeCitationSnippet.supporting_text}"
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Footer Controls for Assistant (Dev Mode Drawer Trigger) */}
                {msg.role === "assistant" && msg.developer_mode && (
                  <div className="mt-3.5 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs">
                    <span className="text-slate-500 font-mono text-[11px]">
                      Latency: {msg.developer_mode.total_latency_ms?.toFixed(0)}ms (Retrieval:{" "}
                      {msg.developer_mode.retrieval_latency_ms?.toFixed(0)}ms)
                    </span>
                    <button
                      onClick={() => onOpenDeveloperDrawer(msg.developer_mode)}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 text-[11px] font-mono transition-colors"
                    >
                      <Terminal className="w-3 h-3" />
                      Inspect Pipeline Trace
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex items-start gap-3">
            <div className="glass-panel-elevated p-4 rounded-2xl flex items-center gap-3 text-xs text-slate-300 font-mono">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span>LangGraph Multi-Agent Pipeline Running (Analyze → Retrieve → Verify)...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form Bar */}
      <div className="p-4 border-t border-slate-800/80 bg-[#0c121e]/80">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex items-center gap-3">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask anything about the uploaded documents (e.g. financial metrics, clauses, architecture)..."
            disabled={isLoading}
            className="flex-1 bg-slate-900/90 border border-slate-800 focus:border-blue-500 rounded-xl px-4 py-3 text-xs sm:text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none transition-all"
          />
          <button
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold tracking-wide disabled:opacity-40 disabled:cursor-not-allowed shadow-glow transition-all flex items-center gap-2"
          >
            <span>Ask</span>
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
};
