#!/usr/bin/env python3
"""
OmniMLX - A lightweight GUI Control Panel for managing local AI servers.
Production release with smart model selection and venv support.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# HuggingFace Hub for model search
try:
    from huggingface_hub import HfApi
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False

# Settings file path
SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Default settings with "pre-installed" feel - HuggingFace IDs auto-download
DEFAULT_SETTINGS = {
    # Model root directories (for dropdown scanning)
    "brain_models_root": "",
    "brain_selected_model": "",
    "ears_models_root": "",
    "ears_selected_model": "",
    "mouth_models_root": "",
    "mouth_selected_model": "",
    "eyes_models_root": "",
    "eyes_selected_model": "",
    # Default HuggingFace model IDs (auto-download on first run)
    "brain_model_path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "ears_model_path": "mlx-community/whisper-large-v3-turbo",
    "mouth_model_path": "mlx-community/Kokoro-82M-bf16",
    "eyes_model_path": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    # Brain settings
    "brain_max_kv_size": 8192,
    # Mouth settings
    "mouth_default_voice": "af_heart",
    "mouth_default_speed": 1.0,
    # Provider settings
    "brain_provider": "local",
    "brain_api_key": "",
    "brain_api_url": "https://api.openai.com/v1",
    "brain_api_model": "gpt-4o",
    "ears_provider": "local",
    "ears_api_key": "",
    "ears_api_url": "https://api.openai.com/v1",
    "ears_api_model": "whisper-1",
    "mouth_provider": "local",
    "mouth_api_key": "",
    "mouth_api_url": "https://api.openai.com/v1",
    "mouth_api_model": "tts-1",
    "eyes_provider": "local",
    "eyes_api_key": "",
    "eyes_api_url": "https://api.openai.com/v1",
    "eyes_api_model": "gpt-4o",
}

# Port configuration
PORTS = {
    "brain": 8080,
    "ears": 8081,
    "mouth": 8082,
    "eyes": 8083
}


def get_python_executable() -> str:
    """
    Get the correct Python executable, preferring venv if available.
    This fixes the 'ModuleNotFoundError' when running from system Python.
    """
    # Check if we're already in a venv
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        return sys.executable
    
    # Check for .venv in script directory
    script_dir = Path(__file__).parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    
    # Check for venv (without dot) in script directory
    venv_python = script_dir / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    
    # Fallback to current executable
    return sys.executable


def scan_models_directory(root_dir: str) -> List[str]:
    """
    Scan a directory for valid model subfolders.
    A valid model folder contains config.json or config.yaml.
    Returns a list of folder names (not full paths).
    """
    if not root_dir or not Path(root_dir).exists():
        return []
    
    models = []
    root_path = Path(root_dir)
    
    try:
        for item in root_path.iterdir():
            if item.is_dir():
                # Check for config files
                config_json = item / "config.json"
                config_yaml = item / "config.yaml"
                config_yml = item / "config.yml"
                
                if config_json.exists() or config_yaml.exists() or config_yml.exists():
                    models.append(item.name)
    except PermissionError:
        pass
    
    return sorted(models)


def load_settings() -> dict:
    """Load settings from JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                # Merge with defaults
                return {**DEFAULT_SETTINGS, **settings}
        except Exception as e:
            print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")


# Default search keywords per tab
DEFAULT_SEARCH_KEYWORDS = {
    "Brain": "Llama",
    "Ears": "Whisper",
    "Mouth": "Kokoro",
    "Eyes": "VL"
}


def search_mlx_models(query: str, limit: int = 15) -> List[dict]:
    """
    Search HuggingFace for MLX models.
    Returns list of {id, downloads, likes, size_gb} sorted by downloads.
    """
    if not HF_API_AVAILABLE:
        return []
    
    try:
        api = HfApi()
        # Search for MLX models - include "mlx" in the query
        search_query = f"{query} mlx" if query else "mlx"
        results = api.list_models(
            search=search_query,
            limit=limit * 2,
            sort="downloads"
        )
        
        # Convert generator to list first
        results_list = list(results)
        
        models = []
        for m in results_list[:limit]:
            # Get model size from siblings (files) metadata
            size_gb = 0.0
            try:
                if hasattr(m, 'siblings') and m.siblings:
                    total_size = sum(f.size or 0 for f in m.siblings if hasattr(f, 'size') and f.size)
                    size_gb = round(total_size / (1024**3), 1)  # Convert bytes to GB
            except:
                pass
            
            models.append({
                "id": m.id,
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
                "size_gb": size_gb
            })
        return models
    except Exception as e:
        print(f"Error searching models: {e}")
        return []


class ModelBrowserDialog(tk.Toplevel):
    """Popup dialog for searching and selecting HuggingFace models."""
    
    def __init__(self, parent, tab_name: str, callback: callable):
        super().__init__(parent)
        self.title(f"🔍 Find MLX Models - {tab_name}")
        self.geometry("600x400")
        self.callback = callback
        self.tab_name = tab_name
        self.models: List[dict] = []
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Build UI
        self._build_ui()
        
        # Auto-search with default keyword
        default_query = DEFAULT_SEARCH_KEYWORDS.get(tab_name, "")
        self.search_entry.insert(0, default_query)
        self._do_search()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self):
        # Search frame
        search_frame = ttk.Frame(self, padding="10")
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        
        search_btn = ttk.Button(search_frame, text="🔍 Search", command=self._do_search)
        search_btn.pack(side=tk.LEFT)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="Results (double-click to select)", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Treeview for results
        columns = ("model_id", "size_gb", "downloads", "likes")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=12)
        self.tree.heading("model_id", text="Model ID")
        self.tree.heading("size_gb", text="💾 Size")
        self.tree.heading("downloads", text="⬇️ Downloads")
        self.tree.heading("likes", text="❤️ Likes")
        self.tree.column("model_id", width=350)
        self.tree.column("size_gb", width=70, anchor="center")
        self.tree.column("downloads", width=90, anchor="center")
        self.tree.column("likes", width=70, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Double-click to select
        self.tree.bind("<Double-1>", self._on_select)
        
        # Button frame
        btn_frame = ttk.Frame(self, padding="10")
        btn_frame.pack(fill=tk.X)
        
        select_btn = ttk.Button(btn_frame, text="✓ Select", command=self._on_select)
        select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.destroy)
        cancel_btn.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(btn_frame, text="")
        self.status_label.pack(side=tk.RIGHT)
    
    def _do_search(self):
        """Execute model search."""
        query = self.search_entry.get().strip()
        if not query:
            return
        
        self.status_label.configure(text="Searching...")
        self.update()
        
        # Clear existing results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Search in background thread to avoid UI freeze
        def search_thread():
            self.models = search_mlx_models(query, limit=20)
            self.after(0, self._update_results)
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def _update_results(self):
        """Update treeview with search results."""
        for model in self.models:
            size_str = f"{model['size_gb']} GB" if model['size_gb'] > 0 else "-"
            self.tree.insert("", tk.END, values=(
                model["id"],
                size_str,
                f"{model['downloads']:,}",
                f"{model['likes']:,}"
            ))
        
        self.status_label.configure(text=f"Found {len(self.models)} models")
    
    def _on_select(self, event=None):
        """Handle model selection."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        model_id = values[0]
        
        # Call callback with selected model ID
        if self.callback:
            self.callback(model_id)
        
        self.destroy()


class ServerTab:
    """Base class for server tabs with Local/External provider switching and smart model selection."""
    
    def __init__(self, parent: ttk.Notebook, name: str, port: int, 
                 server_cmd: Optional[str] = None, extra_widgets: Optional[callable] = None,
                 bridge_type: str = "chat"):
        self.name = name
        self.port = port
        self.server_cmd = server_cmd
        self.bridge_type = bridge_type
        self.process: Optional[subprocess.Popen] = None
        self.log_thread: Optional[threading.Thread] = None
        self.running = False
        self.extra_args: Dict[str, Any] = {}
        
        # Model selection variables
        self.models_root = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.available_models: List[str] = []
        
        # Provider settings
        self.provider = tk.StringVar(value="local")
        self.api_key = tk.StringVar()
        self.api_url = tk.StringVar(value="https://api.openai.com/v1")
        self.api_model = tk.StringVar()
        
        # Create frame
        self.frame = ttk.Frame(parent, padding="10")
        parent.add(self.frame, text=name)
        
        # Build UI
        self._build_ui(extra_widgets)
        
    def _build_ui(self, extra_widgets: Optional[callable] = None):
        # Provider selector (top)
        provider_frame = ttk.LabelFrame(self.frame, text="Provider", padding="5")
        provider_frame.pack(fill=tk.X, pady=(0, 5))
        
        local_rb = ttk.Radiobutton(
            provider_frame, 
            text="🖥️ Local MLX", 
            variable=self.provider,
            value="local",
            command=self._on_provider_change
        )
        local_rb.pack(side=tk.LEFT, padx=(0, 20))
        
        external_rb = ttk.Radiobutton(
            provider_frame, 
            text="☁️ External API", 
            variable=self.provider,
            value="external",
            command=self._on_provider_change
        )
        external_rb.pack(side=tk.LEFT)
        
        # Local model selector frame (with smart dropdown)
        self.local_frame = ttk.LabelFrame(self.frame, text="Local Model", padding="5")
        self.local_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Row 1: Models Root Directory
        root_row = ttk.Frame(self.local_frame)
        root_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(root_row, text="Root Dir:", width=10).pack(side=tk.LEFT)
        self.root_entry = ttk.Entry(root_row, textvariable=self.models_root, width=40)
        self.root_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_root_btn = ttk.Button(root_row, text="Browse...", command=self._browse_root, width=10)
        browse_root_btn.pack(side=tk.LEFT)
        
        # Row 2: Model Selection Dropdown
        model_row = ttk.Frame(self.local_frame)
        model_row.pack(fill=tk.X)
        
        ttk.Label(model_row, text="Model:", width=10).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_row, 
            textvariable=self.selected_model,
            values=self.available_models,
            state="normal",
            width=38
        )
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_select)
        
        refresh_btn = ttk.Button(model_row, text="↻", width=3, command=self._refresh_models)
        refresh_btn.pack(side=tk.LEFT)
        
        # HuggingFace model browser button (always visible)
        hf_btn = ttk.Button(model_row, text="🔍", width=3, command=self._open_model_browser)
        hf_btn.pack(side=tk.LEFT, padx=(2, 0))
        
        # Build remaining UI (external frame, status, logs)
        self._build_remaining_ui(extra_widgets)
    
    def _open_model_browser(self):
        """Open HuggingFace model browser dialog."""
        if not HF_API_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Please run ./setup.sh to install search dependencies.\n\n"
                "Or manually install: pip install huggingface_hub"
            )
            return
        
        def on_model_selected(model_id: str):
            # Set the selected model ID directly (HuggingFace repo ID)
            self.selected_model.set(model_id)
            # Clear the root directory since we're using HF ID directly
            self.models_root.set("")
            self.model_combo.set(model_id)
            self._save_settings()
            self.log(f"Selected from HuggingFace: {model_id}")
        
        ModelBrowserDialog(self.frame.winfo_toplevel(), self.name, on_model_selected)
    
    def _build_remaining_ui(self, extra_widgets):
        """Build remaining UI components after model browser."""
        # External API frame (hidden by default)
        self.external_frame = ttk.LabelFrame(self.frame, text="External API Settings", padding="5")
        
        # API Key
        api_key_row = ttk.Frame(self.external_frame)
        api_key_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(api_key_row, text="API Key:", width=12).pack(side=tk.LEFT)
        self.api_key_entry = ttk.Entry(api_key_row, textvariable=self.api_key, width=50, show="*")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Base URL
        api_url_row = ttk.Frame(self.external_frame)
        api_url_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(api_url_row, text="Base URL:", width=12).pack(side=tk.LEFT)
        self.api_url_entry = ttk.Entry(api_url_row, textvariable=self.api_url, width=50)
        self.api_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Model Name
        api_model_row = ttk.Frame(self.external_frame)
        api_model_row.pack(fill=tk.X)
        
        ttk.Label(api_model_row, text="Model:", width=12).pack(side=tk.LEFT)
        self.api_model_entry = ttk.Entry(api_model_row, textvariable=self.api_model, width=50)
        self.api_model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Extra widgets (e.g., context limit buttons for Brain tab)
        self.extra_widget_frame = ttk.Frame(self.frame)
        self.extra_widget_frame.pack(fill=tk.X, pady=(0, 5))
        if extra_widgets:
            extra_widgets(self.extra_widget_frame)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicator
        indicator_frame = ttk.Frame(status_frame)
        indicator_frame.pack(fill=tk.X)
        
        ttk.Label(indicator_frame, text="Server Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(indicator_frame, text="Stopped", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # Port display
        ttk.Label(indicator_frame, text="URL:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=f"http://127.0.0.1:{self.port}")
        port_entry = ttk.Entry(indicator_frame, textvariable=self.port_var, width=25, state="readonly")
        port_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons
        btn_frame = ttk.Frame(status_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self.start_server)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Logs
        log_frame = ttk.LabelFrame(self.frame, text="Server Logs", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize provider state
        self._on_provider_change()
    
    def _browse_root(self):
        """Browse for models root directory."""
        folder = filedialog.askdirectory(title=f"Select {self.name} Models Root Directory")
        if folder:
            self.models_root.set(folder)
            self._refresh_models()
            self._save_settings()
    
    def _refresh_models(self):
        """Refresh the model dropdown based on current root directory."""
        root = self.models_root.get()
        self.available_models = scan_models_directory(root)
        self.model_combo['values'] = self.available_models
        
        if self.available_models:
            self.model_combo.config(state="readonly")
            # Select first model if none selected
            if not self.selected_model.get() or self.selected_model.get() not in self.available_models:
                self.selected_model.set(self.available_models[0])
            self.log(f"Found {len(self.available_models)} models in {root}")
        else:
            self.model_combo.config(state="disabled")
            self.log(f"No valid models found in {root}")
    
    def _on_model_select(self, event=None):
        """Handle model selection from dropdown."""
        self._save_settings()
        self.log(f"Selected model: {self.selected_model.get()}")
    
    def get_model_path(self) -> str:
        """Get the full model path (combines root + selected model)."""
        root = self.models_root.get()
        model = self.selected_model.get()
        if root and model:
            return str(Path(root) / model)
        # Fallback to direct path from settings
        settings = load_settings()
        return settings.get(f"{self.name.lower()}_model_path", "")
    
    def _on_provider_change(self):
        """Handle provider selection change."""
        if self.provider.get() == "local":
            # Show local frame, hide external frame
            self.external_frame.pack_forget()
            self.local_frame.pack(fill=tk.X, pady=(0, 5))
            # Show extra widgets (context frame, voice frame, etc.)
            self.extra_widget_frame.pack(fill=tk.X, pady=(0, 5))
        else:
            # Show external frame, hide local frame
            self.local_frame.pack_forget()
            self.external_frame.pack(fill=tk.X, pady=(0, 5))
            # Hide extra widgets in external mode
            self.extra_widget_frame.pack_forget()
        
        self._save_settings()
    
    def _save_settings(self):
        """Save current settings to JSON file."""
        settings = load_settings()
        name_lower = self.name.lower()
        settings[f"{name_lower}_models_root"] = self.models_root.get()
        settings[f"{name_lower}_selected_model"] = self.selected_model.get()
        settings[f"{name_lower}_provider"] = self.provider.get()
        settings[f"{name_lower}_api_key"] = self.api_key.get()
        settings[f"{name_lower}_api_url"] = self.api_url.get()
        settings[f"{name_lower}_api_model"] = self.api_model.get()
        if name_lower == "brain" and "brain_max_kv_size" in self.extra_args:
            settings["brain_max_kv_size"] = self.extra_args["brain_max_kv_size"]
        save_settings(settings)
    
    def _load_provider_settings(self):
        """Load provider-specific settings."""
        settings = load_settings()
        name_lower = self.name.lower()
        
        # Load model root and selection
        self.models_root.set(settings.get(f"{name_lower}_models_root", ""))
        
        # Set model ID: prefer selected_model, then default from settings
        model_id = settings.get(f"{name_lower}_selected_model", "")
        if not model_id:
            # Use the default HuggingFace ID from settings
            model_id = settings.get(f"{name_lower}_model_path", "")
        self.selected_model.set(model_id)
        
        # Refresh models dropdown
        if self.models_root.get():
            self._refresh_models()
        
        # Load provider settings
        self.provider.set(settings.get(f"{name_lower}_provider", "local"))
        self.api_key.set(settings.get(f"{name_lower}_api_key", ""))
        self.api_url.set(settings.get(f"{name_lower}_api_url", "https://api.openai.com/v1"))
        self.api_model.set(settings.get(f"{name_lower}_api_model", ""))
        self._on_provider_change()
    
    def log(self, message: str):
        """Append message to log area (thread-safe)."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def start_server(self):
        """Start the server subprocess."""
        # Validate based on provider
        if self.provider.get() == "local":
            model_path = self.get_model_path()
            if not model_path:
                messagebox.showwarning("No Model", "Please select a models root directory and model.")
                return
        else:
            if not self.api_key.get():
                messagebox.showwarning("No API Key", "Please enter an API key for external API.")
                return
            if not self.api_model.get():
                messagebox.showwarning("No Model", "Please enter a model name for external API.")
                return
        
        if self.server_cmd is None and self.provider.get() == "local":
            messagebox.showinfo("Placeholder", f"{self.name} server is not implemented yet.")
            return
        
        # Build command
        cmd = self._build_command()
        if cmd is None:
            return
        
        provider_name = "Local MLX" if self.provider.get() == "local" else "External API"
        self.log(f"Starting {self.name} server ({provider_name})...")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            self.running = True
            
            # Update UI
            self.status_label.configure(text="Running", foreground="green")
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self._disable_inputs()
            
            # Start log reader thread
            self.log_thread = threading.Thread(target=self._read_output, daemon=True)
            self.log_thread.start()
            
        except Exception as e:
            self.log(f"Error starting server: {e}")
            messagebox.showerror("Error", f"Failed to start server: {e}")
    
    def _disable_inputs(self):
        """Disable all input fields while running."""
        self.root_entry.configure(state=tk.DISABLED)
        self.model_combo.configure(state=tk.DISABLED)
        self.api_key_entry.configure(state=tk.DISABLED)
        self.api_url_entry.configure(state=tk.DISABLED)
        self.api_model_entry.configure(state=tk.DISABLED)
    
    def _enable_inputs(self):
        """Re-enable all input fields when stopped."""
        self.root_entry.configure(state=tk.NORMAL)
        if self.available_models:
            self.model_combo.configure(state="readonly")
        self.api_key_entry.configure(state=tk.NORMAL)
        self.api_url_entry.configure(state=tk.NORMAL)
        self.api_model_entry.configure(state=tk.NORMAL)
    
    def _build_command(self) -> Optional[list]:
        """Build the server command based on provider."""
        if self.provider.get() == "external":
            return self._build_bridge_command()
        else:
            return self._build_local_command()
    
    def _build_local_command(self) -> Optional[list]:
        """Build local server command. Override in subclasses."""
        return None
    
    def _build_bridge_command(self) -> Optional[list]:
        """Build bridge_server.py command for external API."""
        script_dir = Path(__file__).parent
        bridge_script = script_dir / "bridge_server.py"
        python_exe = get_python_executable()
        
        return [
            python_exe, str(bridge_script),
            "--port", str(self.port),
            "--type", self.bridge_type,
            "--target_url", self.api_url.get(),
            "--api_key", self.api_key.get(),
            "--model", self.api_model.get()
        ]
    
    def stop_server(self):
        """Stop the server subprocess."""
        if self.process:
            self.log(f"Stopping {self.name} server...")
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                pass
            finally:
                self.process = None
                self.running = False
            
            self.log("Server stopped.")
        
        # Update UI
        self.status_label.configure(text="Stopped", foreground="red")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._enable_inputs()
    
    def _read_output(self):
        """Read subprocess output in background thread."""
        if self.process and self.process.stdout:
            try:
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        break
                    if line:
                        # Schedule UI update on main thread
                        self.frame.after(0, lambda l=line.strip(): self.log(l))
            except Exception as e:
                self.frame.after(0, lambda: self.log(f"Output reader error: {e}"))
    
    def cleanup(self):
        """Clean up resources when closing."""
        self.stop_server()


class BrainTab(ServerTab):
    """Brain tab for LLM (mlx_lm.server)."""
    
    def __init__(self, parent: ttk.Notebook):
        self.context_sizes = {
            "4k": 4096,
            "8k": 8192,
            "16k": 16384,
            "32k": 32768,
            "64k": 65536,
            "128k": 131072
        }
        self.selected_context = tk.StringVar(value="8k")
        super().__init__(
            parent, 
            name="Brain", 
            port=PORTS["brain"],
            server_cmd="mlx_lm.server",
            extra_widgets=self._add_context_buttons,
            bridge_type="chat"
        )
        
        # Load saved context size
        settings = load_settings()
        if "brain_max_kv_size" in settings:
            for label, size in self.context_sizes.items():
                if size == settings["brain_max_kv_size"]:
                    self.selected_context.set(label)
                    self.extra_args["brain_max_kv_size"] = size
                    break
        
        # Load provider settings
        self._load_provider_settings()
    
    def _add_context_buttons(self, frame: ttk.Frame):
        """Add context limit buttons."""
        self.context_frame = ttk.LabelFrame(frame, text="Context Limit", padding="5")
        self.context_frame.pack(fill=tk.X, pady=(0, 10))
        
        for label in self.context_sizes:
            btn = ttk.Radiobutton(
                self.context_frame, 
                text=label.upper(), 
                variable=self.selected_context,
                value=label,
                command=self._on_context_change
            )
            btn.pack(side=tk.LEFT, padx=5)
    
    def _on_context_change(self):
        """Handle context size change."""
        label = self.selected_context.get()
        size = self.context_sizes[label]
        self.extra_args["brain_max_kv_size"] = size
        self._save_settings()
        self.log(f"Context limit set to {label} ({size} tokens)")
    
    def _build_local_command(self) -> Optional[list]:
        """Build mlx_lm.server command for local MLX."""
        kv_size = self.extra_args.get("brain_max_kv_size", 8192)
        model_path = self.get_model_path()
        python_exe = get_python_executable()
        
        return [
            python_exe, "-m", "mlx_lm.server",
            "--model", model_path,
            "--port", str(self.port),
            "--max-kv-size", str(kv_size)
        ]


class EarsTab(ServerTab):
    """Ears tab for Whisper STT."""
    
    def __init__(self, parent: ttk.Notebook):
        super().__init__(
            parent,
            name="Ears",
            port=PORTS["ears"],
            server_cmd="ears_server.py",
            bridge_type="stt"
        )
        
        # Load provider settings
        self._load_provider_settings()
        
        self.log("Whisper STT server - Ready")
        self.log("Endpoint: POST /v1/audio/transcriptions")
    
    def _build_local_command(self) -> Optional[list]:
        """Build ears_server.py command for local MLX."""
        script_dir = Path(__file__).parent
        server_script = script_dir / "ears_server.py"
        model_path = self.get_model_path()
        python_exe = get_python_executable()
        
        return [
            python_exe, str(server_script),
            "--model", model_path,
            "--port", str(self.port)
        ]


class MouthTab(ServerTab):
    """Mouth tab for TTS (Kokoro)."""
    
    # Available voices
    VOICES = [
        ("af_heart", "Pebble (Default warm)"),
        ("af_bella", "Bella (Soft, gentle)"),
        ("af_nicole", "Nicole (Professional)"),
        ("af_sarah", "Sarah (Friendly)"),
        ("af_sky", "Emily (Energetic)"),
        ("am_michael", "Michael (Male, deep)"),
        ("am_adam", "Adam (Male, neutral)"),
        ("am_eric", "Eric (Male, casual)"),
        ("am_liam", "Liam (Male, warm)"),
        ("am_onyx", "Onyx (Male, bold)"),
    ]
    
    def __init__(self, parent: ttk.Notebook):
        self.selected_voice = tk.StringVar(value="af_heart")
        self.speed = tk.DoubleVar(value=1.0)
        super().__init__(
            parent,
            name="Mouth",
            port=PORTS["mouth"],
            server_cmd="mouth_server.py",
            extra_widgets=self._add_voice_selector,
            bridge_type="tts"
        )
        
        # Load saved voice preference
        settings = load_settings()
        if "mouth_default_voice" in settings:
            self.selected_voice.set(settings["mouth_default_voice"])
        if "mouth_default_speed" in settings:
            self.speed.set(settings["mouth_default_speed"])
        
        # Load provider settings
        self._load_provider_settings()
        
        self.log("Kokoro TTS server - Ready")
        self.log("Endpoint: POST /v1/audio/speech")
        self.log(f"Default voice: {self.selected_voice.get()}")
    
    def _add_voice_selector(self, frame: ttk.Frame):
        """Add voice dropdown and speed slider."""
        self.voice_frame = ttk.LabelFrame(frame, text="Voice Settings", padding="5")
        self.voice_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Voice dropdown
        voice_row = ttk.Frame(self.voice_frame)
        voice_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(voice_row, text="Default Voice:").pack(side=tk.LEFT)
        voice_combo = ttk.Combobox(
            voice_row,
            textvariable=self.selected_voice,
            values=[v[0] for v in self.VOICES],
            state="readonly",
            width=15
        )
        voice_combo.pack(side=tk.LEFT, padx=(5, 0))
        voice_combo.bind("<<ComboboxSelected>>", self._on_voice_change)
        
        # Speed slider
        speed_row = ttk.Frame(self.voice_frame)
        speed_row.pack(fill=tk.X)
        
        ttk.Label(speed_row, text="Speed:").pack(side=tk.LEFT)
        speed_scale = ttk.Scale(
            speed_row,
            from_=0.5,
            to=2.0,
            variable=self.speed,
            orient=tk.HORIZONTAL,
            command=self._on_speed_change
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.speed_label = ttk.Label(speed_row, text="1.0x")
        self.speed_label.pack(side=tk.LEFT)
    
    def _on_voice_change(self, event=None):
        """Handle voice selection change."""
        settings = load_settings()
        settings["mouth_default_voice"] = self.selected_voice.get()
        save_settings(settings)
        self.log(f"Default voice set to: {self.selected_voice.get()}")
    
    def _on_speed_change(self, value=None):
        """Handle speed slider change."""
        speed_val = round(self.speed.get(), 1)
        self.speed_label.configure(text=f"{speed_val}x")
        settings = load_settings()
        settings["mouth_default_speed"] = speed_val
        save_settings(settings)
    
    def _build_local_command(self) -> Optional[list]:
        """Build mouth_server.py command for local MLX."""
        script_dir = Path(__file__).parent
        server_script = script_dir / "mouth_server.py"
        model_path = self.get_model_path()
        python_exe = get_python_executable()
        
        return [
            python_exe, str(server_script),
            "--model", model_path,
            "--port", str(self.port)
        ]
    
    def _save_settings(self):
        """Save current settings to JSON file."""
        settings = load_settings()
        name_lower = self.name.lower()
        settings[f"{name_lower}_models_root"] = self.models_root.get()
        settings[f"{name_lower}_selected_model"] = self.selected_model.get()
        settings[f"{name_lower}_provider"] = self.provider.get()
        settings[f"{name_lower}_api_key"] = self.api_key.get()
        settings[f"{name_lower}_api_url"] = self.api_url.get()
        settings[f"{name_lower}_api_model"] = self.api_model.get()
        settings["mouth_default_voice"] = self.selected_voice.get()
        settings["mouth_default_speed"] = self.speed.get()
        save_settings(settings)


class EyesTab(ServerTab):
    """Eyes tab for Vision (mlx_vlm.server)."""
    
    def __init__(self, parent: ttk.Notebook):
        super().__init__(
            parent,
            name="Eyes",
            port=PORTS["eyes"],
            server_cmd="mlx_vlm.server",
            bridge_type="vision"
        )
        
        # Load provider settings
        self._load_provider_settings()
    
    def _build_local_command(self) -> Optional[list]:
        """Build mlx_vlm.server command for local MLX."""
        model_path = self.get_model_path()
        python_exe = get_python_executable()
        
        return [
            python_exe, "-m", "mlx_vlm.server",
            "--model", model_path,
            "--port", str(self.port)
        ]


class ModelsTab:
    """Dedicated tab for browsing and searching HuggingFace MLX models."""
    
    # Model categories with keywords
    CATEGORIES = {
        "All": "",
        "Brain (LLM)": "Llama",
        "Ears (STT)": "Whisper",
        "Mouth (TTS)": "Kokoro",
        "Eyes (Vision)": "VL"
    }
    
    def __init__(self, parent: ttk.Notebook, tabs: Dict[str, ServerTab]):
        self.frame = ttk.Frame(parent, padding="10")
        parent.add(self.frame, text="🔍 Models")
        self.tabs = tabs
        self.models: List[dict] = []
        
        self._build_ui()
    
    def _build_ui(self):
        # Search frame
        search_frame = ttk.LabelFrame(self.frame, text="Search HuggingFace for MLX Models", padding="10")
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Category filter
        cat_row = ttk.Frame(search_frame)
        cat_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(cat_row, text="Category:").pack(side=tk.LEFT)
        self.category_var = tk.StringVar(value="All")
        cat_combo = ttk.Combobox(
            cat_row,
            textvariable=self.category_var,
            values=list(self.CATEGORIES.keys()),
            state="readonly",
            width=15
        )
        cat_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # Search entry
        ttk.Label(cat_row, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(cat_row, textvariable=self.search_var, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        
        search_btn = ttk.Button(cat_row, text="🔍 Search", command=self._do_search)
        search_btn.pack(side=tk.LEFT)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.frame, text="Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview
        columns = ("model_id", "size_gb", "downloads", "likes")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        self.tree.heading("model_id", text="Model ID (double-click to use)")
        self.tree.heading("size_gb", text="💾 Size")
        self.tree.heading("downloads", text="⬇️ Downloads")
        self.tree.heading("likes", text="❤️ Likes")
        self.tree.column("model_id", width=400)
        self.tree.column("size_gb", width=70, anchor="center")
        self.tree.column("downloads", width=90, anchor="center")
        self.tree.column("likes", width=70, anchor="center")
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Double-click to select
        self.tree.bind("<Double-1>", self._on_double_click)
        
        # Action buttons frame
        action_frame = ttk.Frame(self.frame)
        action_frame.pack(fill=tk.X)
        
        # Quick action buttons
        ttk.Label(action_frame, text="Assign to:").pack(side=tk.LEFT)
        
        ttk.Button(action_frame, text="🧠 Brain", command=lambda: self._assign_to("brain")).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="👂 Ears", command=lambda: self._assign_to("ears")).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗣️ Mouth", command=lambda: self._assign_to("mouth")).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="👁️ Eyes", command=lambda: self._assign_to("eyes")).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="📋 Copy ID", command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=20)
        
        # Status label
        self.status_label = ttk.Label(action_frame, text="")
        self.status_label.pack(side=tk.RIGHT)
    
    def _do_search(self):
        """Execute model search."""
        if not HF_API_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Please run ./setup.sh to install search dependencies.\n\n"
                "Or manually install: pip install huggingface_hub"
            )
            return
        
        # Build query from category + search term
        category = self.category_var.get()
        cat_keyword = self.CATEGORIES.get(category, "")
        search_term = self.search_var.get().strip()
        
        # Combine category keyword with search term
        if cat_keyword and search_term:
            query = f"{cat_keyword} {search_term}"
        elif cat_keyword:
            query = cat_keyword
        else:
            query = search_term if search_term else "mlx"
        
        self.status_label.configure(text="Searching...")
        self.frame.update()
        
        # Clear results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Search in background thread
        def search_thread():
            self.models = search_mlx_models(query, limit=30)
            self.frame.after(0, self._update_results)
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def _update_results(self):
        """Update treeview with search results."""
        for model in self.models:
            size_str = f"{model['size_gb']} GB" if model['size_gb'] > 0 else "-"
            self.tree.insert("", tk.END, values=(
                model["id"],
                size_str,
                f"{model['downloads']:,}",
                f"{model['likes']:,}"
            ))
        
        self.status_label.configure(text=f"Found {len(self.models)} models")
    
    def _get_selected_model(self) -> Optional[str]:
        """Get the currently selected model ID."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Select Model", "Please select a model from the list first.")
            return None
        
        item = selection[0]
        values = self.tree.item(item, "values")
        return values[0]
    
    def _assign_to(self, tab_name: str):
        """Assign selected model to a specific tab."""
        model_id = self._get_selected_model()
        if not model_id:
            return
        
        # Get the target tab
        tab = self.tabs.get(tab_name)
        if not tab:
            return
        
        # Update the tab's model selection
        tab.selected_model.set(model_id)
        tab.models_root.set("")  # Clear root since we're using HF ID
        tab._save_settings()
        tab.log(f"Model set to: {model_id}")
        
        self.status_label.configure(text=f"✓ Assigned to {tab_name.title()}")
    
    def _copy_to_clipboard(self):
        """Copy selected model ID to clipboard."""
        model_id = self._get_selected_model()
        if not model_id:
            return
        
        self.frame.clipboard_clear()
        self.frame.clipboard_append(model_id)
        self.status_label.configure(text=f"✓ Copied: {model_id}")
    
    def _on_double_click(self, event=None):
        """Handle double-click - show assignment menu."""
        model_id = self._get_selected_model()
        if model_id:
            # Ask which tab to assign to
            result = messagebox.askyesno(
                "Assign Model",
                f"Assign {model_id} to Brain tab?\n\n"
                f"Click 'No' to see other options.",
                icon="question"
            )
            if result:
                self._assign_to("brain")


class OmniMLXApp:
    """Main application class."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OmniMLX Control Panel")
        self.root.geometry("650x580")
        self.root.minsize(550, 450)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create server tabs
        self.tabs = {}
        
        # Brain tab
        self.tabs["brain"] = BrainTab(self.notebook)
        
        # Ears tab
        self.tabs["ears"] = EarsTab(self.notebook)
        
        # Mouth tab
        self.tabs["mouth"] = MouthTab(self.notebook)
        
        # Eyes tab
        self.tabs["eyes"] = EyesTab(self.notebook)
        
        # Models tab (HuggingFace browser)
        self.models_tab = ModelsTab(self.notebook, self.tabs)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Style
        self._configure_style()
    
    def _configure_style(self):
        """Configure ttk style."""
        style = ttk.Style()
        style.configure("TLabelframe", padding=5)
        style.configure("TButton", padding=5)
    
    def _on_close(self):
        """Handle window close event."""
        # Stop all servers
        for tab in self.tabs.values():
            tab.cleanup()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = OmniMLXApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()