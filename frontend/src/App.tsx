import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { DocumentManager } from "./components/DocumentManager";
import { ChatInterface } from "./components/ChatInterface";
import { DeveloperDrawer } from "./components/DeveloperDrawer";
import { EvaluationModal } from "./components/EvaluationModal";
import { api } from "./services/api";
import { DocumentItem, ChatMessage, DeveloperMetrics } from "./types";

export function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [devMode, setDevMode] = useState(true);
  const [activeDevMetrics, setActiveDevMetrics] = useState<DeveloperMetrics | null>(null);
  const [isEvaluationOpen, setIsEvaluationOpen] = useState(false);
  const [activeTenant, setActiveTenant] = useState("default_tenant");
  const [systemStatus, setSystemStatus] = useState<{
    status: string;
    services: Record<string, string>;
  }>({
    status: "healthy",
    services: { qdrant: "connected", redis: "connected" },
  });

  useEffect(() => {
    // Initial health check and documents load
    const init = async () => {
      const health = await api.checkHealth();
      setSystemStatus(health);
      const docs = await api.listDocuments();
      setDocuments(docs);
    };
    init();
  }, []);

  const handleTenantChange = (tenant: string) => {
    setActiveTenant(tenant);
    api.setTenant(tenant);
    api.listDocuments().then(setDocuments);
  };

  const handleToggleDocSelect = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleSelectAll = () => {
    setSelectedDocIds(documents.map((d) => d.id));
  };

  const handleClearSelection = () => {
    setSelectedDocIds([]);
  };

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    try {
      const newDoc = await api.uploadDocument(file);
      setDocuments((prev) => [newDoc, ...prev]);
    } catch (err: any) {
      alert(`Upload error: ${err.message || "Failed to process document"}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    const ok = await api.deleteDocument(docId);
    if (ok) {
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
    }
  };

  const handleSendMessage = async (text: string) => {
    const userMsg: ChatMessage = {
      id: "msg_" + Date.now(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    const assistantMsgId = "msg_asst_" + Date.now();
    let currentAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    };

    try {
      const response = await api.query(
        text,
        selectedDocIds,
        undefined,
        (tokenContent) => {
          // Streaming partial update
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.id === assistantMsgId) {
              return [
                ...prev.slice(0, -1),
                { ...last, content: tokenContent },
              ];
            } else {
              return [
                ...prev,
                { ...currentAssistantMsg, content: tokenContent },
              ];
            }
          });
        }
      );

      // Final complete message with audit & citations
      const completedMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: response.answer,
        timestamp: new Date().toISOString(),
        citations: response.citations,
        confidence_score: response.confidence_score,
        grounded: response.grounded,
        refused: response.refused,
        refusal_reason: response.refusal_reason,
        audit: response.audit,
        developer_mode: response.developer_mode,
      };

      setMessages((prev) => {
        const withoutAssistant = prev.filter((m) => m.id !== assistantMsgId);
        return [...withoutAssistant, completedMsg];
      });

      if (devMode && response.developer_mode) {
        setActiveDevMetrics(response.developer_mode);
      }
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: `Error: ${err.message || "An unexpected error occurred during processing."}`,
        timestamp: new Date().toISOString(),
        refused: true,
        refusal_reason: "System failure communicating with pipeline.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#080c14] text-slate-100 overflow-hidden font-sans">
      {/* Top Header */}
      <Header
        systemStatus={systemStatus}
        devMode={devMode}
        onToggleDevMode={() => setDevMode(!devMode)}
        onOpenEvaluation={() => setIsEvaluationOpen(true)}
        activeTenant={activeTenant}
        onTenantChange={handleTenantChange}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Document Manager */}
        <DocumentManager
          documents={documents}
          selectedDocIds={selectedDocIds}
          onToggleDocSelect={handleToggleDocSelect}
          onSelectAll={handleSelectAll}
          onClearSelection={handleClearSelection}
          onUpload={handleUpload}
          onDelete={handleDelete}
          isUploading={isUploading}
        />

        {/* Center: Evidence Chat Interface */}
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          onOpenDeveloperDrawer={(metrics) => setActiveDevMetrics(metrics)}
          selectedDocCount={selectedDocIds.length}
        />
      </div>

      {/* Developer Drawer Inspector */}
      <DeveloperDrawer
        metrics={activeDevMetrics}
        isOpen={Boolean(activeDevMetrics)}
        onClose={() => setActiveDevMetrics(null)}
      />

      {/* Evaluation Benchmark Modal */}
      <EvaluationModal
        isOpen={isEvaluationOpen}
        onClose={() => setIsEvaluationOpen(false)}
      />
    </div>
  );
}

export default App;
