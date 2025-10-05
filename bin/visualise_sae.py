"""
Interactive exploration of different functions coming from sparse autoencoders.

Almost all of this script (basically everything shiny) is AI generated.
"""

import base64
from glob import glob
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastcore.script import call_parse
from PIL import Image
from scipy.stats import gaussian_kde
from shiny import App, reactive, render, ui

from metalign.data import h5_to_np


def image_to_base64(img_path, max_size=200):
    with Image.open(img_path) as img:
        img.thumbnail((max_size, max_size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

def create_image_grid(image_paths, title):
    if not image_paths: return f"<h3>{title}</h3><p>No images found</p>"
    images_html = ""
    for img_path in image_paths:
        img_base64 = image_to_base64(img_path)
        img_name = Path(img_path).stem
        images_html += f"""
        <div style="display: inline-block; margin: 5px; text-align: center;">
            <img src="{img_base64}" style="max-width: 150px; max-height: 150px;"/>
            <br><small>{img_name}</small>
        </div>
        """
    return f"""
    <div>
        <h3>{title}</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            {images_html}
        </div>
    </div>
    """

def create_kde_plot(feature_values, feature_dim):
    nonzero_values = feature_values[feature_values != 0]
    if len(nonzero_values) < 2: return ""
    
    plt.figure(figsize=(10, 4))
    kde = gaussian_kde(nonzero_values)
    x_range = np.linspace(nonzero_values.min(), nonzero_values.max(), 1000)
    plt.plot(x_range, kde(x_range), 'b-', linewidth=2)
    plt.fill_between(x_range, kde(x_range), alpha=0.3)
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.title(f'Feature {feature_dim} Distribution (Non-zero values only)')
    plt.grid(True, alpha=0.3)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    plot_str = base64.b64encode(buffer.getvalue()).decode()
    return f"<img src='data:image/png;base64,{plot_str}' style='max-width: 100%;'/>"

def load_annotation(feature_dim):
    annotation_path = Path(f"data/sae/annotations/{model_name}_feat{feature_dim}_nonzero{min_nonzero}.txt")
    return annotation_path.read_text().strip() if annotation_path.exists() else ""

def save_annotation(feature_dim, description):
    annotation_path = Path(f"data/sae/annotations/{model_name}_feat{feature_dim}_nonzero{min_nonzero}.txt")
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(description)

min_nonzero = 100 
image_paths = sorted(glob("data/external/THINGS/*/*jpg"))  # Adjust this path as needed
model_name = "things_sae-top_k-64-cls_only-layer_11-hook_resid_post"
features = h5_to_np(model_name=model_name, min_nonzero=min_nonzero)

if len(image_paths) != features.shape[0]: print(f"Warning: {len(image_paths)} images but {features.shape[0]} feature rows")

app_ui = ui.page_fluid(
    ui.h1("Interactive Feature Visualization"),
    ui.row(
        ui.column(3, ui.input_numeric("feature_dim", "Feature Dimension:", value=0, min=0, max=features.shape[1]-1)),
        ui.column(3, ui.input_numeric("k_images", "Number of images:", value=6, min=1, max=20)),
        ui.column(3, ui.input_numeric("threshold", "Threshold:", value=0, step=0.01)),
        ui.column(3, ui.input_switch("randomize", "Randomize selection", value=False))
    ),
    ui.div(ui.h3("Feature Statistics"), ui.output_text("feature_stats"), style="margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"),
    ui.row(
        ui.column(8, ui.input_text_area("description", "Feature Description:", placeholder="Enter your description of what this feature represents...", rows=3)),
        ui.column(4, ui.br(), ui.input_action_button("save_description", "Save Description", class_="btn-success"))
    ),
    ui.output_ui("current_annotation"),
    ui.output_ui("image_display"),
    ui.output_ui("kde_plot")
)

def server(input, output, session):
    
    @reactive.calc
    def get_feature_data():
        dim = int(input.feature_dim())
        threshold = input.threshold()
        feature_values = features[:, dim]
        nonzero_mask = feature_values != 0
        zero_mask = feature_values == 0
        above_threshold_mask = feature_values > threshold
        nonzero_indices = np.where(nonzero_mask)[0]
        zero_indices = np.where(zero_mask)[0]
        above_threshold_indices = np.where(above_threshold_mask)[0]
        return {'values': feature_values, 'nonzero_indices': nonzero_indices, 'zero_indices': zero_indices, 'above_threshold_indices': above_threshold_indices, 'n_zeros': np.sum(zero_mask), 'n_above_threshold': np.sum(above_threshold_mask), 'total': len(feature_values)}
    
    @reactive.effect
    @reactive.event(input.save_description)
    def save_current_description():
        dim = int(input.feature_dim())
        description = input.description()
        save_annotation(dim, description)
        ui.update_text_area("description", value="")
        ui.notification_show(f"Description saved for feature {dim}!", type="success")
    
    @render.text
    def feature_stats():
        data = get_feature_data()
        fraction_zeros = data['n_zeros'] / data['total']
        fraction_above_threshold = data['n_above_threshold'] / data['total']
        return f"Feature {input.feature_dim()} of {features.shape[1]} total features. {data['n_zeros']} images ({fraction_zeros:.1%}) have value 0. {data['n_above_threshold']} images ({fraction_above_threshold:.1%}) above threshold {input.threshold()}"
    
    @render.ui
    def current_annotation():
        dim = int(input.feature_dim())
        annotation = load_annotation(dim)
        return ui.div(ui.h4("Current Annotation:"), ui.p(annotation, style="font-style: italic; color: #666;"), style="margin: 10px 0; padding: 10px; background-color: #e8f4f8; border-radius: 5px;") if annotation else ui.div()
    
    @render.ui
    def image_display():
        data = get_feature_data()
        k = input.k_images()
        randomize = input.randomize()
        threshold = input.threshold()
        
        if threshold > 0:
            target_indices = data['above_threshold_indices']
            selected_nonzero = np.random.choice(target_indices, min(k, len(target_indices)), replace=False) if randomize and len(target_indices) >= k else target_indices[:k] if randomize else target_indices[np.argsort(data['values'][target_indices])[-k:][::-1]]
        else:
            target_indices = data['nonzero_indices']
            selected_nonzero = np.random.choice(target_indices, min(k, len(target_indices)), replace=False) if randomize and len(target_indices) >= k else target_indices[:k] if randomize else target_indices[np.argsort(data['values'][target_indices])[-k:][::-1]]
        
        selected_zero = np.random.choice(data['zero_indices'], min(k, len(data['zero_indices'])), replace=False) if len(data['zero_indices']) >= k else data['zero_indices']
        nonzero_paths = [image_paths[i] for i in selected_nonzero]
        zero_paths = [image_paths[i] for i in selected_zero]
        title1 = f"{'Random' if randomize else 'Top'} {k} {'Above Threshold' if threshold > 0 else 'Non-Zero'} Images"
        title2 = f"Random {len(selected_zero)} Zero-Value Images"
        html_content = create_image_grid(nonzero_paths, title1) + "<hr style='margin: 30px 0;'/>" + create_image_grid(zero_paths, title2)
        return ui.HTML(html_content)
    
    @render.ui
    def kde_plot():
        data = get_feature_data()
        kde_html = create_kde_plot(data['values'], input.feature_dim())
        return ui.HTML(f"<div style='margin: 30px 0;'><h3>Feature Distribution</h3>{kde_html}</div>") if kde_html else ui.div()

app = App(app_ui, server)

@call_parse
def main():
    app.run()