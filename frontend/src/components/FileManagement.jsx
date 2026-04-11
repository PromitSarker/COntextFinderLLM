import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Trash2, AlertTriangle, Loader2, CheckCircle2, FileText, Globe, RefreshCw } from 'lucide-react';

function FileManagement({ API_BASE }) {
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await axios.get(`${API_BASE}/documents`);
      setDocuments(res.data);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Are you sure you want to delete ALL documents? This cannot be undone.")) return;

    setLoading(true);
    try {
      const res = await axios.delete(`${API_BASE}/documents/all`);
      setStatus({ type: 'success', msg: res.data.message });
      fetchDocuments();
    } catch (err) {
      setStatus({ type: 'error', msg: "Failed to delete all documents." });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFile = async (filename) => {
    if (!window.confirm(`Delete "${filename}"?`)) return;

    setLoading(true);
    try {
      const res = await axios.delete(`${API_BASE}/document/${encodeURIComponent(filename)}`);
      setStatus({ type: 'success', msg: res.data.message });
      fetchDocuments();
    } catch (err) {
      setStatus({ type: 'error', msg: `Failed to delete ${filename}` });
    } finally {
      setLoading(false);
    }
  };



  return (
    <div className="manage-container">
      <section className="glass card animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <div className="card-header">
          <FileText className="text-primary" />
          <h3>Indexed Documents</h3>
          <button className="icon-btn refresh-btn" onClick={fetchDocuments} title="Refresh List">
            <RefreshCw size={18} className={loadingDocs ? 'animate-spin' : ''} />
          </button>
        </div>
        <p className="text-muted">Currently indexed files and web pages in your library.</p>
        
        <div className="doc-list-container">
          {loadingDocs ? (
            <div className="loading-state">
              <Loader2 className="animate-spin" />
              <span>Loading documents...</span>
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-state">
              <p>No documents indexed yet.</p>
            </div>
          ) : (
            <div className="doc-grid">
              {documents.map((doc, idx) => (
                <div key={idx} className="doc-item glass">
                  <div className="doc-icon">
                    {doc.type === 'web_url' ? <Globe size={20} /> : <FileText size={20} />}
                  </div>
                  <div className="doc-info">
                    <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                    <span className="doc-meta">{doc.chunk_count} chunks indexed</span>
                  </div>
                  <button 
                    className="delete-item-btn" 
                    onClick={() => handleDeleteFile(doc.filename)}
                    disabled={loading}
                    title="Delete document"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="glass card animate-fade-in" style={{ marginTop: '2rem' }}>
        <div className="card-header">
          <AlertTriangle className="text-error" />
          <h3>Danger Zone</h3>
        </div>
        <p className="text-muted">Manage your knowledge base by clearing data.</p>
        
        <div className="manage-actions">
          <div className="action-row">
            <div className="action-info">
              <h4>Delete All Data</h4>
              <p>Wipe the entire vector database and all uploaded files.</p>
            </div>
            <button 
              className="btn-danger" 
              onClick={handleDeleteAll}
              disabled={loading}
            >
              {loading ? <Loader2 className="animate-spin" /> : <><Trash2 size={18} /> Delete All</>}
            </button>
          </div>
        </div>
      </section>

      {status && (
        <div className={`status-banner glass animate-fade-in ${status.type}`}>
          {status.type === 'success' ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
          <span>{status.msg}</span>
          <button onClick={() => setStatus(null)} className="close-btn">×</button>
        </div>
      )}

      <style jsx>{`
        .card-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }

        .refresh-btn {
          margin-left: auto;
          background: transparent;
          color: var(--text-muted);
        }

        .refresh-btn:hover {
          color: var(--primary);
        }

        .doc-list-container {
          margin-top: 1.5rem;
          min-height: 200px;
        }

        .loading-state, .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          height: 200px;
          color: var(--text-muted);
        }

        .doc-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1rem;
        }

        .doc-item {
          padding: 1rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          transition: border-color 0.2s;
        }

        .doc-item:hover {
          border-color: var(--primary);
        }

        .doc-icon {
          color: var(--primary);
          background: rgba(139, 92, 246, 0.1);
          padding: 0.75rem;
          border-radius: 0.5rem;
        }

        .doc-info {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-width: 0;
        }

        .doc-name {
          font-weight: 500;
          font-size: 0.95rem;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          color: var(--text-main);
        }

        .doc-meta {
          font-size: 0.8rem;
          color: var(--text-muted);
        }

        .delete-item-btn {
          background: transparent;
          color: var(--text-muted);
          padding: 0.5rem;
        }

        .delete-item-btn:hover {
          color: var(--error);
          background: rgba(239, 68, 68, 0.1);
        }

        .manage-actions {
          margin-top: 2rem;
        }

        .action-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 2rem;
        }

        .action-info h4 {
          margin-bottom: 0.25rem;
        }

        .action-info p {
          font-size: 0.9rem;
          color: var(--text-muted);
        }

        .btn-danger {
          background: rgba(239, 68, 68, 0.1);
          color: var(--error);
          border: 1px solid rgba(239, 68, 68, 0.2);
          padding: 0.75rem 1.5rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .btn-danger:hover {
          background: var(--error);
          color: white;
        }





        .status-banner {
          position: fixed;
          bottom: 2rem;
          right: 2rem;
          padding: 1rem 1.5rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          border-radius: 0.75rem;
          z-index: 100;
        }

        .status-banner.success {
          color: var(--success);
          border-color: var(--success);
        }

        .status-banner.error {
          color: var(--error);
          border-color: var(--error);
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

export default FileManagement;
