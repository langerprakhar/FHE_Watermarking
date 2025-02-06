import numpy as np
from concrete import fhe
from PIL import Image
import gradio as gr
import time
import os
import shutil
import binascii
from pathlib import Path

class FHEWatermarking:
    def __init__(self):
        self.fhe_circuit = None
        self.client = None
        self.server = None
        self.IMAGE_SIZE = 64
        self.key_dir = Path("keys")
        self.filter_name = "watermark"
        self.filter_path = Path("filters") / self.filter_name / "deployment"

    def apply_watermark(self, x):
        """
        Apply watermark pattern
        x: input image array of shape (64, 64)
        """
        # Simple diagonal pattern that works with FHE
        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                if i == j:
                    x[i, j] = 255
        return x

    def compile_model(self):
        try:
            print("Creating inputset...")
            inputset = [
                np.random.randint(0, 256, size=(self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.int64)
                for _ in range(2)
            ]

            print("Initializing compiler...")
            compiler = fhe.Compiler(
                self.apply_watermark,
                {"x": "encrypted"}
            )

            print("Compiling FHE circuit...")
            self.fhe_circuit = compiler.compile(
                inputset,
                show_mlir=True
            )
            print("Compilation complete!")

            # Save the circuit for the client and server
            self.save_circuit()

            # Initialize the client and server
            self.client = FHEClient(self.filter_path, self.filter_name, self.key_dir)
            self.server = FHEServer(self.filter_path)

        except Exception as e:
            print(f"Error during compilation: {e}")
            raise

    def save_circuit(self):
        """
        Save the compiled FHE circuit for the client and server.
        """
        self.filter_path.mkdir(parents=True, exist_ok=True)
        self.fhe_circuit.server.save(self.filter_path / "server.zip", via_mlir=True)
        self.fhe_circuit.client.save(self.filter_path / "client.zip")

    def generate_private_key(self):
        """
        Generate the private key for FHE.
        """
        self.client.generate_private_and_evaluation_keys(force=True)

    def encrypt_image(self, input_array):
        """
        Encrypt the input image using FHE.
        """
        return self.client.encrypt_serialize(input_array)

    def decrypt_image(self, encrypted_output):
        """
        Decrypt the encrypted image using the correct private key.
        """
        return self.client.deserialize_decrypt_post_process(encrypted_output)

    def decrypt_with_incorrect_key(self, encrypted_output):
        """
        Decrypt the encrypted image using an incorrect private key.
        """
        wrong_client = FHEClient(self.filter_path, self.filter_name)
        wrong_client.generate_private_and_evaluation_keys(force=True)
        return wrong_client.deserialize_decrypt_post_process(encrypted_output)

    def process_image(self, input_image, progress=gr.Progress()):
        if input_image is None:
            return None

        try:
            # Pre-processing
            progress(0.2, desc="Pre-processing image...")
            img_gray = Image.fromarray(input_image).convert('L')
            img_small = img_gray.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            input_array = np.array(img_small, dtype=np.int64)

            # Encryption
            progress(0.4, desc="Encrypting...")
            encrypted_input = self.encrypt_image(input_array)

            # Get REAL encrypted data
            encrypted_bytes = encrypted_input
            encrypted_data = np.frombuffer(encrypted_bytes, dtype=np.uint8)
            # Reshape the actual encrypted data for visualization
            encrypted_height = int(np.sqrt(len(encrypted_data)))
            encrypted_visual = encrypted_data[:encrypted_height*encrypted_height].reshape(encrypted_height, encrypted_height)

            # FHE Processing
            progress(0.6, desc="Applying watermark in encrypted domain...")
            encrypted_output = self.server.run(encrypted_input, self.client.get_serialized_evaluation_keys())

            # Get encrypted output visualization
            output_bytes = encrypted_output
            output_data = np.frombuffer(output_bytes, dtype=np.uint8)
            output_height = int(np.sqrt(len(output_data)))
            output_visual = output_data[:output_height*output_height].reshape(output_height, output_height)

            # Decryption with incorrect key (for visualization)
            progress(0.7, desc="Decrypting with incorrect key...")
            incorrect_decrypted_output = self.decrypt_with_incorrect_key(encrypted_output)
            incorrect_final_image = Image.fromarray(incorrect_decrypted_output.astype(np.uint8)).resize((256, 256))

            # Decryption with correct key
            progress(0.8, desc="Decrypting result...")
            decrypted_output = self.decrypt_image(encrypted_output)

            # Final result
            final_image = Image.fromarray(decrypted_output.astype(np.uint8)).resize((256, 256))

            results = {
                'original': input_image,
                'grayscale': np.array(img_gray.resize((256, 256))),
                'encrypted': encrypted_visual,
                'encrypted_processed': output_visual,
                'incorrect_decrypted': np.array(incorrect_final_image),
                'decrypted': np.array(final_image),
                'technical_details': {
                    'Original Size': f"{input_image.shape}",
                    'Encrypted Size': f"{encrypted_visual.shape}",
                    'Encryption Ratio': f"{len(encrypted_bytes) / input_array.nbytes:.2f}x",
                    'Encrypted Data (First 100 bytes)': binascii.hexlify(encrypted_bytes[:100]).decode('ascii')
                }
            }

            progress(1.0, desc="Complete!")
            return results

        except Exception as e:
            print(f"Error in processing: {e}")
            return None

class FHEClient:
    """Client interface to encrypt and decrypt FHE data associated to a Filter."""

    def __init__(self, path_dir, filter_name, key_dir=None):
        """Initialize the FHE interface.
        Args:
            path_dir (Path): The path to the directory where the circuit is saved.
            filter_name (str): The filter's name to consider.
            key_dir (Path): The path to the directory where the keys are stored. Default to None.
        """
        self.path_dir = path_dir
        self.key_dir = key_dir

        # If path_dir does not exist raise
        assert path_dir.exists(), f"{path_dir} does not exist. Please specify a valid path."

        # Load the client
        self.client = fhe.Client.load(self.path_dir / "client.zip", self.key_dir)

    def generate_private_and_evaluation_keys(self, force=False):
        """Generate the private and evaluation keys.
        Args:
            force (bool): If True, regenerate the keys even if they already exist.
        """
        self.client.keygen(force)

    def get_serialized_evaluation_keys(self):
        """Get the serialized evaluation keys.
        Returns:
            bytes: The evaluation keys.
        """
        return self.client.evaluation_keys.serialize()

    def encrypt_serialize(self, input_image):
        """Encrypt and serialize the input image in the clear.
        Args:
            input_image (numpy.ndarray): The image to encrypt and serialize.
        Returns:
            bytes: The pre-processed, encrypted and serialized image.
        """
        # Encrypt the image
        encrypted_image = self.client.encrypt(input_image)

        # Serialize the encrypted image to be sent to the server
        serialized_encrypted_image = encrypted_image.serialize()
        return serialized_encrypted_image

    def deserialize_decrypt_post_process(self, serialized_encrypted_output_image):
        """Deserialize, decrypt and post-process the output image in the clear.
        Args:
            serialized_encrypted_output_image (bytes): The serialized and encrypted output image.
        Returns:
            numpy.ndarray: The decrypted, deserialized and post-processed image.
        """
        # Deserialize the encrypted image
        encrypted_output_image = fhe.Value.deserialize(
            serialized_encrypted_output_image
        )

        # Decrypt the image
        output_image = self.client.decrypt(encrypted_output_image)

        return output_image

class FHEServer:
    """Server interface to run a FHE circuit."""

    def __init__(self, path_dir):
        """Initialize the FHE interface.
        Args:
            path_dir (Path): The path to the directory where the circuit is saved.
        """
        self.path_dir = path_dir

        # Load the FHE circuit
        self.server = fhe.Server.load(self.path_dir / "server.zip")

    def run(self, serialized_encrypted_image, serialized_evaluation_keys):
        """Run the filter on the server over an encrypted image.
        Args:
            serialized_encrypted_image (bytes): The encrypted and serialized image.
            serialized_evaluation_keys (bytes): The serialized evaluation keys.
        Returns:
            bytes: The filter's output.
        """
        # Deserialize the encrypted input image and the evaluation keys
        encrypted_image = fhe.Value.deserialize(serialized_encrypted_image)
        evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)

        # Execute the filter in FHE
        encrypted_output = self.server.run(encrypted_image, evaluation_keys=evaluation_keys)

        # Serialize the encrypted output image
        serialized_encrypted_output = encrypted_output.serialize()

        return serialized_encrypted_output

def create_interface():
    watermarker = FHEWatermarking()
    print("Initializing FHE model...")
    watermarker.compile_model()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # FHE Watermarking Demonstration
        This demo shows the complete process of applying a watermark to an image using Fully Homomorphic Encryption (FHE).
        The watermark is applied while the image remains encrypted.
        """)

        with gr.Column():
            # Input Section
            input_image = gr.Image(
                type="numpy",
                label="Upload Image",
                scale=1,
                height=256,
                width=256
            )
            process_btn = gr.Button("▶️ Start Processing", scale=1)

            # Processing Steps Display
            gr.Markdown("### Processing Steps")

            with gr.Row():
                original_display = gr.Image(
                    label="1. Original Image",
                    height=256,
                    width=256
                )
                grayscale_display = gr.Image(
                    label="2. Grayscale Conversion",
                    height=256,
                    width=256
                )

            gr.Markdown("### Final Result")
            with gr.Row():
                incorrect_decrypted_display = gr.Image(
                    label="3. Incorrectly Decrypted Image",
                    height=256,
                    width=256
                )
                final_display = gr.Image(
                    label="4. Decrypted Result with Watermark",
                    height=256,
                    width=256
                )

            with gr.Accordion("Technical Details", open=False):
                technical_info = gr.JSON()

        def process_and_update(image):
            if image is None:
                return [None] * 5

            results = watermarker.process_image(image)
            if results is None:
                return [None] * 5

            return [
                results['original'],
                results['grayscale'],
                results['incorrect_decrypted'],
                results['decrypted'],
                results['technical_details']
            ]

        process_btn.click(
            fn=process_and_update,
            inputs=[input_image],
            outputs=[
                original_display,
                grayscale_display,
                incorrect_decrypted_display,
                final_display,
                technical_info
            ]
        )

    return demo

if __name__ == "__main__":
    try:
        demo = create_interface()
        demo.launch(server_port=7860)
    except Exception as e:
        print(f"Fatal error: {e}")
