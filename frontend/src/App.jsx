import React, { useState } from 'react';
import { Search as SearchIcon, Upload as UploadIcon, Settings, Database, BrainCircuit } from 'lucide-react';
import Search from './components/Search';
import Upload from './components/Upload';
import FileManagement from './components/FileManagement';

const API_BASE = window.location.origin === 'http://localhost:5173' 
  ? "http://localhost:2000" 
  : window.location.origin;

function App() {
  const [activeTab, setActiveTab] = useState('search');

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar glass">
        <div className="logo-area">
          <BrainCircuit size={32} className="logo-icon" />
          <h2>ContextFinder</h2>
        </div>
        
        <nav>
          <button 
            className={`nav-item ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <SearchIcon size={20} />
            <span>Search</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <UploadIcon size={20} />
            <span>Upload</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'manage' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage')}
          >
            <Database size={20} />
            <span>Manage</span>
          </button>
        </nav>

        <div className="status-footer">
          <div className="status-dot online"></div>
          <span>Backend Connected</span>
        </div>
      </aside>

      {/* Main Content */}
      <main className="content-area">
        <header className="main-header">
          <h1>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</h1>
          <div className="header-actions">
            <button className="icon-btn"><Settings size={20} /></button>
          </div>
        </header>

        <div className="container animate-fade-in">
          {activeTab === 'search' && <Search API_BASE={API_BASE} />}
          {activeTab === 'upload' && <Upload API_BASE={API_BASE} />}
          {activeTab === 'manage' && <FileManagement API_BASE={API_BASE} />}
        </div>
      </main>

      <style jsx>{`
        .app-layout {
          display: flex;
          height: 100vh;
        }

        .sidebar {
          width: 280px;
          height: 100%;
          border-radius: 0;
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          padding: 2rem 1.5rem;
        }

        .logo-area {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-bottom: 3rem;
        }

        .logo-icon {
          color: var(--primary);
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: 1rem;
          width: 100%;
          padding: 1rem;
          margin-bottom: 0.5rem;
          background: transparent;
          color: var(--text-muted);
          text-align: left;
          font-size: 1rem;
        }

        .nav-item:hover {
          color: var(--text-main);
          background: rgba(255,255,255,0.05);
        }

        .nav-item.active {
          background: rgba(139, 92, 246, 0.15);
          color: var(--primary);
          border-left: 3px solid var(--primary);
        }

        .status-footer {
          margin-top: auto;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.8rem;
          color: var(--text-muted);
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-dot.online {
          background: var(--success);
          box-shadow: 0 0 8px var(--success);
        }

        .content-area {
          flex: 1;
          overflow-y: auto;
          background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.05), transparent);
        }

        .main-header {
          padding: 1.5rem 2rem;
          border-bottom: 1px solid var(--border);
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .header-actions .icon-btn {
          background: transparent;
          color: var(--text-muted);
          padding: 0.5rem;
        }

        .header-actions .icon-btn:hover {
          color: var(--text-main);
        }
      `}</style>
    </div>
  );
}

export default App;
