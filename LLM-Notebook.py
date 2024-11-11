import os
import json
import requests
import gradio as gr
import tiktoken
from datetime import datetime
from typing import Iterator, Tuple, Dict, List
from pathlib import Path

# Default settings
default_settings = {
    "port": 7860,
    "default_model": "meta-llama/llama-3.1-70b-instruct:free",
    "theme": "default",
    "api_key": "",
    "use_auth": False,
    "username": "admin",
    "password": "password",
    "launch_in_browser": False
}

# Default sampler preset
default_preset = {
    "temperature": 0.7,
    "top_p": 1.0,
    "min_p": 0.05,
    "top_k": 40,
    "max_tokens": 512
}

def load_todo_list() -> Dict[str, str]:
    """Load todo list from file."""
    todo_path = "todo.json"
    if os.path.exists(todo_path):
        with open(todo_path, "r") as f:
            return json.load(f)
    return {}

def load_sampler_presets():
    """Load sampler presets from file."""
    presets_path = "sampler_presets.json"
    if os.path.exists(presets_path):
        with open(presets_path, "r") as f:
            return json.load(f)
    else:
        presets = {"Default": default_preset}
        with open(presets_path, "w") as f:
            json.dump(presets, f, indent=4)
        return presets

def save_sampler_preset(name: str, temperature: float, top_p: float, min_p: float, 
                       top_k: int, max_tokens: int) -> str:
    """Save a sampler preset."""
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "max_tokens": max_tokens
    }
    presets = load_sampler_presets()
    presets[name] = params
    with open("sampler_presets.json", "w") as f:
        json.dump(presets, f, indent=4)
    return f"Preset '{name}' saved successfully."

def load_preset_values(preset_name: str) -> Dict:
    """Load values for a specific preset."""
    presets = load_sampler_presets()
    return presets.get(preset_name, default_preset)

def ensure_exports_dir():
    """Ensure exports directory exists."""
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    return exports_dir

def load_settings():
    settings_path = "user_settings.json"
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            settings = json.load(f)
            for key, value in default_settings.items():
                if key not in settings:
                    settings[key] = value
            if settings.get("default_model"):
                settings["default_model"] = settings["default_model"].lower().replace(" ", "-")
            return settings
    else:
        with open(settings_path, "w") as f:
            json.dump(default_settings, f, indent=4)
        print(f"Created default settings file: {settings_path}")
        return default_settings.copy()

def save_settings(new_settings):
    settings_path = "user_settings.json"
    with open(settings_path, "w") as f:
        json.dump(new_settings, f, indent=4)

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def format_cost(price_per_token: float) -> str:
    """Format the cost for display per million tokens."""
    price_per_million = price_per_token * 1_000_000
    return f"${price_per_million:.2f}/1M tokens"

def normalize_model_id(model_id: str) -> str:
    """Normalize model ID format."""
    return model_id.lower().replace(" ", "-").strip()

def get_model_info_display(model_info: Dict) -> str:
    """Format model information for display."""
    prompt_price = format_cost(model_info["pricing"]["prompt"])
    completion_price = format_cost(model_info["pricing"]["completion"])
    context_length = model_info.get("context_length", "Unknown")
    return f"Input: {prompt_price} | Output: {completion_price} | Context: {context_length} tokens"

def fetch_models(api_key) -> List[Dict]:
    port = user_settings.get('port', 7860)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": f"http://localhost:{port}",
        "X-Title": "LLM-Notebook"
    }

    try:
        response = requests.get(MODELS_URL, headers=headers)
        response.raise_for_status()

        models_info = response.json()
        models = models_info.get("data", [])
        if not models:
            print("No models found. Using default.")
            default_model = {
                "id": user_settings["default_model"],
                "name": "Llama 3.1 70B Instruct",
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096
            }
            return [default_model]

        model_list = []
        for model in models:
            model_id = normalize_model_id(model["id"])
            model_name = model.get("name", model_id)
            prompt_price = float(model.get("pricing", {}).get("prompt", 0.0))
            completion_price = float(model.get("pricing", {}).get("completion", 0.0))
            context_length = model.get("context_length", 4096)
            model_list.append({
                "id": model_id,
                "name": model_name,
                "pricing": {
                    "prompt": prompt_price,
                    "completion": completion_price
                },
                "context_length": context_length
            })
        return model_list

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"General error occurred: {e}")
    
    return [{"id": user_settings["default_model"], "name": "Llama 3.1 70B Instruct", 
             "pricing": {"prompt": 0.0, "completion": 0.0}, "context_length": 4096}]

def save_user_settings(port, default_model, theme, api_key, launch_in_browser):
    global user_settings
    user_settings.update({
        "port": int(port),
        "default_model": model_choices.get(default_model, default_model),
        "theme": theme,
        "api_key": api_key,
        "launch_in_browser": launch_in_browser
    })

    save_settings(user_settings)
    return "Settings saved successfully. Please restart the application for changes to take effect."

def export_chat(text: str) -> Tuple[str, str]:
    """Export chat content to a JSON file."""
    try:
        exports_dir = ensure_exports_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = exports_dir / f"chat_export_{timestamp}.json"
        
        export_data = {
            "timestamp": timestamp,
            "content": text,
            "token_count": count_tokens(text)
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(filename), f"Chat exported to {filename}"
    except Exception as e:
        return "", f"Error exporting chat: {str(e)}"

# Global variable to control generation
should_stop = False

def set_stop_flag():
    global should_stop
    should_stop = True
    return "Generation stopped."

def reset_stop_flag():
    global should_stop
    should_stop = False

def process_chunk(chunk: str) -> str:
    """Process a chunk of the API response."""
    if chunk.startswith("data: "):
        chunk = chunk[6:]
    if chunk == "[DONE]":
        return ""
    try:
        data = json.loads(chunk)
        return data.get('choices', [{}])[0].get('delta', {}).get('content', '')
    except json.JSONDecodeError:
        return ""

def update_token_counter(text: str) -> str:
    """Update token counter display."""
    count = count_tokens(text)
    return f"Tokens: {count}"

def update_model_info(model_name: str) -> str:
    """Update model information display."""
    model_info = next((m for m in models_info if m["id"] == model_choices[model_name]), None)
    if model_info:
        return get_model_info_display(model_info)
    return "Model information unavailable"

def generate_text(prompt: str, model_name: str, max_tokens: int, temperature: float, 
                 top_p: float, min_p: float, top_k: int) -> Iterator[Tuple[str, str, str]]:
    """Generate text with streaming responses."""
    global should_stop
    reset_stop_flag()
    
    port = user_settings.get('port', 7860)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {user_settings.get('api_key')}",
        "HTTP-Referer": f"http://localhost:{port}",
        "X-Title": "LLM-Notebook"
    }

    model_id = model_choices[model_name]
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "top_k": int(top_k),
        "stream": True
    }

    try:
        with requests.post(API_URL, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            current_text = prompt
            
            for line in response.iter_lines(decode_unicode=True):
                if should_stop:
                    return current_text, "Generation stopped.", update_token_counter(current_text)
                    
                if line:
                    content = process_chunk(line)
                    if content:
                        current_text += content
                        yield current_text, "Generating...", update_token_counter(current_text)

            yield current_text, "Generation completed.", update_token_counter(current_text)

    except requests.exceptions.RequestException as e:
        error_response = e.response.text if e.response is not None else str(e)
        print("Error Response from API:", error_response)
        yield prompt, f"Failed to generate text: {error_response}", update_token_counter(prompt)

def launch_interface():
    global model_choices, models_info
    
    models_info = fetch_models(user_settings.get("api_key", ""))
    model_choices = {model["name"]: model["id"] for model in models_info}
    reverse_model_choices = {v: k for k, v in model_choices.items()}
    
    default_model_id = normalize_model_id(user_settings.get("default_model", ""))
    default_model_name = reverse_model_choices.get(
        default_model_id,
        next(iter(model_choices.keys()))
    )

    presets = load_sampler_presets()
    todo_list = load_todo_list()

    # Create the interface
    with gr.Blocks(theme=user_settings.get("theme", 'default')) as iface:
        # Add custom CSS
        gr.HTML("""
            <style>
            .generating {
                background: none !important;
            }
            .no-container {
                border: none !important;
                box-shadow: none !important;
            }
            .no-loading {
                transition: none !important;
                border: none !important;
                box-shadow: none !important;
            }
            .no-loading.generating {
                background: none !important;
                opacity: 1 !important;
            }
            </style>
        """)

        with gr.Tab("Notebook"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_output = gr.Textbox(
                        lines=20,
                        label="Collaborative Writing Space",
                        placeholder="Start writing here or paste your text...",
                        interactive=True,
                        show_copy_button=True,
                        show_label=True,
                        elem_classes=["no-container"]
                    )
                with gr.Column(scale=1):
                    token_counter = gr.Textbox(
                        label="Token Count",
                        value="Tokens: 0",
                        interactive=False,
                        elem_classes=["no-loading"]
                    )
            
            status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    model_dropdown = gr.Dropdown(
                        choices=list(model_choices.keys()),
                        value=default_model_name,
                        label="Model",
                        elem_classes=["no-container"]
                    )
                    model_info = gr.Textbox(
                        label="Model Information",
                        value=update_model_info(default_model_name),
                        interactive=False,
                        elem_classes=["no-loading"]
                    )
                with gr.Column(scale=2):
                    generate_btn = gr.Button("Continue Writing", variant="primary")
                    stop_btn = gr.Button("Stop Generation", variant="stop")
                    export_btn = gr.Button("Export Chat", variant="secondary")

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(presets.keys()),
                        value="Default",
                        label="Sampler Preset"
                    )
                    preset_name = gr.Textbox(
                        label="New Preset Name",
                        placeholder="Enter name to save current settings as preset"
                    )
                    load_preset_btn = gr.Button("Load Preset")
                    save_preset_btn = gr.Button("Save As Preset")

                with gr.Row():
                    temperature_slider = gr.Slider(minimum=0.0, maximum=3.0, value=0.7, label="Temperature")
                    top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Top P")
                    min_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, label="Min P")
                    top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top K")
                    tokens_slider = gr.Slider(minimum=1, maximum=8192, step=1, value=512, label="Max Tokens")

        # Settings Tab
        with gr.Tab("Settings"): 
            port_input = gr.Number(value=user_settings.get("port", 7860), label="Port")
            default_model_input = gr.Dropdown(choices=list(model_choices.keys()), value=default_model_name, label="Default Model")
            theme_input = gr.Dropdown(choices=["default", "dark", "light"], value=user_settings.get("theme"), label="Theme")
            api_key_input = gr.Textbox(value=user_settings.get("api_key"), type="password", label="API Key")
            launch_in_browser_checkbox = gr.Checkbox(value=user_settings.get('launch_in_browser', False), label="Launch in Browser")
            
            save_settings_btn = gr.Button("Save Settings")
            save_status = gr.Textbox(label="Save Status", interactive=False)

            save_settings_btn.click(
                fn=save_user_settings,
                inputs=[port_input, default_model_input, theme_input, api_key_input, launch_in_browser_checkbox],
                outputs=[save_status]
            )

        # Improvements Tab
        with gr.Tab("Improvements"):
            for header, text in todo_list.items():
                with gr.Row():
                    gr.Markdown(f"### {header}")
                    gr.Markdown(text)

        # Event handlers
        def update_counters(text):
            return update_token_counter(text)

        prompt_output.change(
            fn=update_counters,
            inputs=[prompt_output],
            outputs=[token_counter],
            queue=False
        )

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )

        # Generate text event
        generate_event = generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_output, model_dropdown, tokens_slider,
                temperature_slider, top_p_slider, min_p_slider, top_k_slider
            ],
            outputs=[prompt_output, status, token_counter],
            show_progress=False,
            queue=True
        )

        stop_btn.click(
            fn=set_stop_flag,
            inputs=[],
            outputs=[status],
            cancels=[generate_event],
            queue=False
        )

        export_btn.click(
            fn=export_chat,
            inputs=[prompt_output],
            outputs=[gr.File(label="Download"), status]
        )

        def load_preset(preset_name):
            values = load_preset_values(preset_name)
            return [
                values["temperature"],
                values["top_p"],
                values["min_p"],
                values["top_k"],
                values["max_tokens"]
            ]

        load_preset_btn.click(
            fn=load_preset,
            inputs=[preset_dropdown],
            outputs=[temperature_slider, top_p_slider, min_p_slider, top_k_slider, tokens_slider]
        )

        save_preset_btn.click(
            fn=save_sampler_preset,
            inputs=[preset_name, temperature_slider, top_p_slider, min_p_slider, top_k_slider, tokens_slider],
            outputs=[status]
        ).then(
            fn=lambda: gr.Dropdown(choices=list(load_sampler_presets().keys())),
            outputs=[preset_dropdown]
        )

    # Launch the Gradio interface with settings
    port = user_settings.get('port', 7860)
    launch_options = {
        'server_name': "127.0.0.1",
        'server_port': port,
        'share': False,
        'inbrowser': user_settings.get('launch_in_browser', False),
    }

    if user_settings.get("use_auth", False):
        launch_options["auth"] = (
            user_settings.get("username", "admin"),
            user_settings.get("password", "password")
        )

    iface.queue().launch(**launch_options)

# Main entry point
if __name__ == "__main__":
    user_settings = load_settings()
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    launch_interface()
