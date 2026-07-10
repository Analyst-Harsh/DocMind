"use client";

import { useState } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { Sidebar } from "@/components/layout/Sidebar";
import { MessageList } from "@/components/chat/MessageList";
import { InputArea } from "@/components/chat/InputArea";
import { useChat } from "@/hooks/useChat";
import { useDocuments } from "@/hooks/useDocuments";

export default function ChatPage() {
  const { messages, isLoading, lastQueryMeta, sendMessage, retryMessage } =
    useChat();
  const {
    documents,
    isLoadingList: isLoadingDocuments,
    isUploading,
    uploadError,
    uploadDocument,
  } = useDocuments();
  const [input, setInput] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    sendMessage(trimmed);
    setInput("");
  };

  const handleQuestionSelect = (q: string) => {
    setInput(q);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopBar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen((prev) => !prev)}
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar
          lastQueryMeta={lastQueryMeta}
          isOpen={isSidebarOpen}
          documents={documents}
          isLoadingDocuments={isLoadingDocuments}
          isUploading={isUploading}
          uploadError={uploadError}
          onUpload={uploadDocument}
        />
        <main className="flex flex-1 flex-col min-h-0">
          <MessageList
            messages={messages}
            onRetry={retryMessage}
            onQuestionSelect={handleQuestionSelect}
          />
          <InputArea
            value={input}
            onChange={setInput}
            onSend={handleSend}
            isLoading={isLoading}
          />
        </main>
      </div>
    </div>
  );
}
