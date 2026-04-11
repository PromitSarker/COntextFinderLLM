import React, { useState } from 'react';
import axios from 'axios';
import { Search as SearchIcon, Loader2, FileText, ExternalLink, Sparkles, Zap, BookOpen, Terminal } from 'lucide-react';

function Search({ API_BASE }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE}/query`, {
        question: query,
        top_k: 6,
        categories: null
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Search failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const WelcomeSection = () => (
    <div className="welcome-section animate-fade-in">
      <div className="welcome-header">
        <Sparkles className="text-primary" size={32} />
        <h2>Welcome to ContextFinder AI</h2>
        <p>Your intelligent document research assistant.</p>
      </div>

      <div className="welcome-grid">
        <div className="welcome-card glass">
          <Zap className="text-accent" size={24} />
          <h3>What it does</h3>
          <p>Semantic search that understands the meaning behind your queries, not just keywords.</p>
        </div>

        <div className="welcome-card glass">
          <BookOpen className="text-primary" size={24} />
          <h3>How it works</h3>
          <p>Upload files or URLs in the <strong>Upload</strong> tab. We chunk and index them using state-of-the-art AI.</p>
        </div>

        <div className="welcome-card glass">
          <Terminal className="text-muted" size={24} />
          <h3>Why use it</h3>
          <p>Get instant answers with precise source citations. Stop digging through thousands of pages manually.</p>
        </div>
      </div>
    </div>
  );



  return (
    <div className="search-container">
      <section className="search-input-area glass card">
        <form onSubmit={handleSearch}>
          <div className="input-row">
            <div className="search-box">
              <SearchIcon className="search-icon" size={20} />
              <input 
                type="text" 
                placeholder="Ask anything about your documents..." 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? <Loader2 className="animate-spin" /> : 'Search'}
            </button>
          </div>


        </form>
      </section>

      {error && <div className="error-msg card glass">{error}</div>}

      {!results && !loading && <WelcomeSection />}

      {results && (
        <div className="results-area">
          {/* AI Answer */}
          <section className="answer-section glass card animate-fade-in">
            <h3>Answer</h3>
            <p className="ai-answer">{results.answer}</p>
          </section>

          {/* Sources */}
          <section className="sources-section">
            <h3>Sources</h3>
            <div className="sources-grid">
              {results.results.map((item, idx) => (
                <div key={idx} className="source-card glass animate-fade-in">
                  <div className="source-header">
                    <FileText size={16} className="text-primary" />
                    <span className="filename">{item.filename}</span>
                    <a href={item.pdf_link} target="_blank" rel="noopener noreferrer" className="ext-link">
                      <ExternalLink size={14} />
                    </a>
                  </div>
                  <p className="source-excerpt">{item.content}</p>
                  <div className="source-footer">
                    <span className="page-tag">Page {item.page_number}</span>
                  </div>
                </div>
              ))}
              {results.results.length === 0 && (
                <p className="text-muted">No relevant source chunks found.</p>
              )}
            </div>
          </section>
        </div>
      )}

      <style jsx>{`
        .input-row {
          display: flex;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .search-box {
          flex: 1;
          position: relative;
          display: flex;
          align-items: center;
        }

        .search-icon {
          position: absolute;
          left: 1rem;
          color: var(--text-muted);
        }

        .search-box input {
          width: 100%;
          padding-left: 3rem;
          font-size: 1.1rem;
          height: 3.5rem;
        }



        .ai-answer {
          line-height: 1.6;
          font-size: 1.1rem;
          color: var(--text-main);
          margin-top: 1rem;
        }

        .sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }

        .source-card {
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.8rem;
        }

        .source-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          color: var(--text-main);
          font-weight: 600;
        }

        .filename {
          flex: 1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .ext-link {
          color: var(--primary);
        }

        .source-excerpt {
          font-size: 0.85rem;
          color: var(--text-muted);
          display: -webkit-box;
          -webkit-line-clamp: 4;
          -webkit-box-orient: vertical;
          overflow: hidden;
          line-height: 1.5;
        }

        .source-footer {
          margin-top: auto;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.75rem;
        }

        .page-tag {
          background: rgba(45, 212, 191, 0.1);
          color: var(--accent);
          padding: 0.2rem 0.5rem;
          border-radius: 4px;
        }



        .error-msg {
          color: var(--error);
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.2);
        }

        .welcome-section {
          margin-top: 3rem;
          text-align: center;
        }

        .welcome-header {
          margin-bottom: 3rem;
        }

        .welcome-header h2 {
          font-size: 2rem;
          margin: 0.5rem 0;
          background: linear-gradient(to right, #fff, var(--text-muted));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .welcome-header p {
          color: var(--text-muted);
          font-size: 1.1rem;
        }

        .welcome-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          max-width: 1000px;
          margin: 0 auto;
        }

        .welcome-card {
          padding: 2rem;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
          transition: transform 0.3s ease;
        }

        .welcome-card:hover {
          transform: translateY(-5px);
          border-color: var(--primary);
        }

        .welcome-card h3 {
          font-size: 1.1rem;
          color: var(--text-main);
        }

        .welcome-card p {
          font-size: 0.9rem;
          color: var(--text-muted);
          line-height: 1.5;
        }

        .animate-spin {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default Search;
