
import { useState } from "react";
import "./App.css";

import { marked } from "marked";
import DOMPurify from "dompurify";

const API_URL = "http://127.0.0.1:8000/chat";

function App() {
  const [userId, setUserId] = useState("1");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  async function sendMessage(text = message) {
    const trimmed = text.trim();

    if (!userId.trim() || !trimmed || loading) return;

    const conversationId =
      sessionStorage.getItem("supportConversationId") ||
      crypto.randomUUID();

    sessionStorage.setItem(
      "supportConversationId",
      conversationId
    );

    setMessages((previous) => [
      ...previous,
      { role: "user", text: trimmed },
    ]);
    setMessage("");
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          message: trimmed,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.answer || "No response was generated.",
        },
      ]);

      setResult(data);
    } catch (error) {
      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text:
            "I couldn't connect to the support server. " +
            "Please check that FastAPI is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">✦</div>
        <div>
          <h1>SupportAI</h1>
          <p>Your intelligent customer support assistant</p>
        </div>
        <span className="online-status">● Ready</span>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <p className="eyebrow">CUSTOMER</p>
          <label htmlFor="userId">Customer ID</label>
          <input
            id="userId"
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
            inputMode="numeric"
            placeholder="Enter customer ID"
          />

          <div className="sidebar-divider" />

          <p className="eyebrow">QUICK QUESTIONS</p>
          {[
            "What is your return policy?",
            "How can I track my order?",
            "What payment methods are supported?",
            "How can I cancel my order?",
          ].map((question) => (
            <button
              className="quick-action"
              key={question}
              onClick={() => sendMessage(question)}
              disabled={loading}
            >
              {question}
            </button>
          ))}

          <div className="sidebar-note">
            <span>✧</span>
            <p>Answers are generated using your customer-support knowledge base.</p>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-heading">
            <div>
              <h2>Customer Support</h2>
              <p>Ask a question about orders, returns, shipping, or payments.</p>
            </div>
            <span className="secure-label">AI ASSISTANT</span>
          </div>

          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="welcome">
                <div className="welcome-icon">✦</div>
                <h2>How can I help you today?</h2>
                <p>
                  Ask me about returns, refunds, order tracking,
                  exchanges, or product information.
                </p>
              </div>
            )}

            {messages.map((item, index) => (
              <div
                className={`message-row ${item.role}`}
                key={`${index}-${item.role}`}
              >
                <div className="message-avatar">
                  {item.role === "assistant" ? "✦" : "You"}
                </div>
                <div className="message-content">
                  <span className="message-name">
                    {item.role === "assistant" ? "SupportAI" : "You"}
                  </span>
                  <div
                    className="message-bubble markdown-content"
                    dangerouslySetInnerHTML={{
                      __html: DOMPurify.sanitize(
                        marked.parse(item.text || "", { breaks: true, async: false })
                      ),
                    }}
                  />
                </div>
              </div>
            ))}

            {loading && (
              <div className="typing-indicator">
                <span />
                <span />
                <span />
                <p>SupportAI is thinking…</p>
              </div>
            )}
          </div>

          {result && (
            <div className="result-strip">
              <span>
                Confidence: {Math.round((result.confidence || 0) * 100)}%
              </span>
              <span>
                {result.escalated ? "Escalated to support" : "AI response"}
              </span>
              {result.ticket_id && (
                <span>Ticket: {result.ticket_id}</span>
              )}
            </div>
          )}

          <form className="composer" onSubmit={handleSubmit}>
            <input
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Type your question here..."
              aria-label="Your message"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !message.trim()}
            >
              {loading ? "..." : "➤"}
            </button>
          </form>

          <p className="disclaimer">
            AI-generated responses may need verification for account-specific issues.
          </p>
        </section>
      </section>
    </main>
  );
}

export default App;
