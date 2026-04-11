import React, { useState } from 'react';
import axios from 'axios';
import { Upload as UploadIcon, Link as LinkIcon, Loader2, CheckCircle2, AlertCircle, X, Globe } from 'lucide-react';

function Upload({ API_BASE }) {
  const [files, setFiles] = useState([]);
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'success' | 'error', msg: string }

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };



  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (files.length === 0) return;

    setLoading(true);
    setStatus(null);
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    
    formData.append('categories', 'default');

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatus({ type: 'success', msg: `Successfully uploaded ${response.data.length} files.` });
      setFiles([]);
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || "Upload failed." });
    } finally {
      setLoading(false);
    }
  };

  const handleUrlUpload = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setStatus(null);
    try {
      const response = await axios.post(`${API_BASE}/upload/url`, null, {
        params: { 
          url: url,
          categories: ['default']
        }
      });
      setStatus({ type: 'success', msg: `Successfully ingested URL: ${url}` });
      setUrl('');
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || "URL ingestion failed." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-grid">
        {/* File Upload Card */}
        <section className="glass card animate-fade-in">
          <div className="card-header">
            <UploadIcon className="text-primary" />
            <h3>Upload Documents</h3>
          </div>
          <p className="text-muted">PDF, Image, DOCX, PPTX, XLSX, TXT, etc.</p>
          
          <form onSubmit={handleFileUpload} className="upload-form">
            <div className="file-dropzone">
              <input 
                type="file" 
                multiple 
                onChange={handleFileChange} 
                id="file-input"
                className="hidden-input"
              />
              <label htmlFor="file-input" className="dropzone-label">
                {files.length > 0 ? (
                  <div className="file-list">
                    {files.map((f, i) => <div key={i} className="file-item">{f.name}</div>)}
                  </div>
                ) : (
                  <>
                    <UploadIcon size={32} />
                    <span>Click to browse files</span>
                  </>
                )}
              </label>
            </div>

            <button 
              type="submit" 
              className="btn-primary w-full" 
              disabled={loading || files.length === 0}
            >
              {loading ? <Loader2 className="animate-spin" /> : 'Upload Files'}
            </button>
          </form>
        </section>

        {/* URL Ingestion Card */}
        <section className="glass card animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <div className="card-header">
            <Globe className="text-accent" />
            <h3>Ingest Website</h3>
          </div>
          <p className="text-muted">Enter a URL to crawl and index its content.</p>
          
          <form onSubmit={handleUrlUpload} className="upload-form">
            <div className="url-input-box">
              <LinkIcon className="url-icon" size={20} />
              <input 
                type="url" 
                placeholder="https://example.com" 
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>
            <button 
              type="submit" 
              className="btn-accent w-full" 
              disabled={loading || !url.trim()}
            >
              {loading ? <Loader2 className="animate-spin" /> : 'Ingest URL'}
            </button>
          </form>
        </section>
      </div>



      {status && (
        <div className={`status-banner glass animate-fade-in ${status.type}`}>
          {status.type === 'success' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
          <span>{status.msg}</span>
          <button onClick={() => setStatus(null)} className="close-btn"><X size={16} /></button>
        </div>
      )}

      <style jsx>{`
        .upload-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .card-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.5rem;
        }

        .upload-form {
          margin-top: 1.5rem;
        }

        .file-dropzone {
          border: 2px dashed var(--border);
          border-radius: 0.75rem;
          padding: 2rem;
          text-align: center;
          margin-bottom: 1rem;
          cursor: pointer;
          transition: border-color 0.2s;
        }

        .file-dropzone:hover {
          border-color: var(--primary);
        }

        .hidden-input {
          display: none;
        }

        .dropzone-label {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
          color: var(--text-muted);
          cursor: pointer;
        }

        .file-list {
          font-size: 0.85rem;
          text-align: left;
          width: 100%;
        }

        .file-item {
          padding: 0.25rem 0;
          color: var(--text-main);
        }

        .url-input-box {
          position: relative;
          display: flex;
          align-items: center;
          margin-bottom: 1rem;
        }

        .url-icon {
          position: absolute;
          left: 1rem;
          color: var(--text-muted);
        }

        .url-input-box input {
          width: 100%;
          padding-left: 3rem;
        }

        .w-full {
          width: 100%;
          height: 3rem;
        }

        .btn-accent {
          background: var(--accent);
          color: #0f172a;
        }

        .btn-accent:hover {
          filter: brightness(1.1);
          box-shadow: 0 4px 12px rgba(45, 212, 191, 0.3);
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

        .close-btn {
          background: transparent;
          color: inherit;
          margin-left: 1rem;
          opacity: 0.7;
        }

        .close-btn:hover {
          opacity: 1;
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

export default Upload;
