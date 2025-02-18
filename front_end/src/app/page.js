'use client'
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]); // Array of { role, content }
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!message.trim()) return;

    // Add the user's message to the conversation
    const userMessage = { role: "user", content: message.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setMessage("");
    setIsLoading(true);
    setError(null);

    try {
      // Use fetch to POST the message with stream mode enabled.
      const response = await fetch("http://127.0.0.1:3006/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content, stream: true }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      // Prepare to read the response stream.
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let aiMessageContent = "";
      let buffer = "";

      // Create an empty AI message so that we can update it as new chunks arrive.
      setMessages((prev) => [...prev, { role: "ai", content: "" }]);

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by a double newline.
        const parts = buffer.split("\n\n");
        // Process every complete SSE event except the last partial one.
        for (let i = 0; i < parts.length - 1; i++) {
          const part = parts[i].trim();
          if (part.startsWith("data: ")) {
            const dataLine = part.substring("data: ".length);
            if (dataLine) {
              aiMessageContent += dataLine;
              // Update the last AI message in the conversation.
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: "ai", content: aiMessageContent };
                return updated;
              });
            }
          }
        }
        // The last part might be incomplete, so we keep it in the buffer.
        buffer = parts[parts.length - 1];
      }
    } catch (err) {
      console.error(err);
      setError("An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 text-center font-bold text-xl">
        AI Chat
      </header>

      {/* Chat Window */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-4">
          {messages.map((msg, index) => {
            const isUser = msg.role === "user";
            return (
              <div
                key={index}
                className={`flex ${isUser ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-xs break-words rounded-lg p-3 shadow-md ${
                    isUser
                      ? "bg-blue-500 text-white"
                      : "bg-gray-200 text-gray-800"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="px-4 py-2 text-center text-red-600">{error}</div>
      )}

      {/* Input Area */}
      <form
        onSubmit={handleSubmit}
        className="p-4 border-t border-gray-300 bg-white"
      >
        <div className="flex items-center">
          <textarea
            className="flex-1 resize-none rounded-full border border-gray-300 p-3 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
            rows="1"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="ml-3 rounded-full bg-blue-600 px-5 py-3 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </main>
  );
}