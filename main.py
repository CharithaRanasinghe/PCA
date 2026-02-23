import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy, skew, kurtosis
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
from PIL import Image

uploader = widgets.FileUpload(accept='image/*', multiple=False)
k_slider = widgets.IntSlider(value=30, min=5, max=100, step=1, description='Components (K):')
output_area = widgets.Output()

def analyze_image(change):
    with output_area:
        clear_output()

        if not uploader.value:
            print("Please upload an image.")
            return

        try:
            content = uploader.value[0]['content']
        except (KeyError, TypeError):
            file_name = list(uploader.value.keys())[0]
            content = uploader.value[file_name]['content']

        img = Image.open(io.BytesIO(content)).convert('RGB')
        img = img.resize((512, 512))
        img_array = np.array(img) / 255.0
        h, w, c = img_array.shape

        residual_features = []

        reconstructed_rgb = np.zeros_like(img_array)

        for i in range(c):
            channel = img_array[:, :, i]
            k = k_slider.value
            pca = PCA(n_components=k)
            transformed = pca.fit_transform(channel)
            reconstructed = pca.inverse_transform(transformed)
            reconstructed_rgb[:, :, i] = reconstructed

            residual = np.abs(channel - reconstructed)
            mean_r = residual.mean()
            std_r = residual.std()
            max_r = residual.max()
            entropy_r = entropy(np.histogram(residual, bins=256, range=(0,1))[0] + 1e-6)
            skew_r = skew(residual.flatten())
            kurt_r = kurtosis(residual.flatten())
            residual_features.extend([mean_r, std_r, max_r, entropy_r, skew_r, kurt_r])

        mean_feat = np.mean(residual_features[0::6])  
        std_feat = np.mean(residual_features[1::6])
        entropy_feat = np.mean(residual_features[3::6])

        ai_score = 0
        ai_score += max(0, (0.2 - std_feat) / 0.2)   
        ai_score += max(0, (0.05 - mean_feat) / 0.05) 
        ai_score += max(0, (1.0 - entropy_feat))      

        
        ai_score = np.clip(ai_score / 3, 0, 1)  
        ai_percentage = ai_score * 100

        if ai_percentage > 50:
            prediction = f"Likely AI-generated ({ai_percentage:.1f}%)"
        else:
            prediction = f"Likely Real Photo ({100-ai_percentage:.1f}%)"

        residual_map = np.sum(np.abs(img_array - reconstructed_rgb), axis=2)

        variance_curve = []
        for i in range(c):
            channel = img_array[:, :, i]
            pca = PCA(n_components=k_slider.value)
            pca.fit(channel)
            variance_curve.append(np.cumsum(pca.explained_variance_ratio_))
        variance_curve = np.mean(variance_curve, axis=0)

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        ax[0].imshow(img_array)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(residual_map, cmap='magma')
        ax[1].set_title(f"Residual Artifact Map (K={k_slider.value})")
        ax[1].axis('off')

        ax[2].plot(variance_curve, color='red', linewidth=2)
        ax[2].set_title("Information Recovery Curve")
        ax[2].set_xlabel("Number of Components (K)")
        ax[2].set_ylabel("Total Variance Captured")
        ax[2].grid(True, alpha=0.3)
        plt.show()

        print("-"*50)
        print(f"PREDICTION: {prediction}")
        print("-"*50)
        print("Residual features (mean, std, entropy over channels):")
        print(f"Mean={mean_feat:.4f}, Std={std_feat:.4f}, Entropy={entropy_feat:.4f}")

uploader.observe(analyze_image, names='value')
k_slider.observe(analyze_image, names='value')

display(widgets.VBox([
    widgets.HTML("<h3>PCA RGB Forensic Microscope with AI Score</h3>"),
    uploader, k_slider, output_area
]))
