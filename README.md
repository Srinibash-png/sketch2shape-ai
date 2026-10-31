<<<<<<< HEAD
# SKETCH2SHAPE AI

### From a 2D Sketch to a 3D Asset

**TechZephyr 4.0 | IIT Bhubaneswar**
**24-Hour Live Hackathon | Computer Vision & Generative AI**

---

## 📌 1. Project Overview

SKETCH2SHAPE AI is an AI-powered application designed to transform a two-dimensional (2D) hand-drawn sketch into a three-dimensional (3D) digital model.

Traditional 3D modelling requires technical knowledge of geometry creation, mesh editing, texturing, lighting, and rendering. This makes it difficult for beginners, designers, and people without professional modelling experience to convert their ideas into 3D objects.

Our project aims to simplify this process by combining Computer Vision, Generative AI, and 3D reconstruction techniques.

Users can provide their ideas through:

* A sketch drawn directly inside the application.
* An existing sketch image uploaded from their device.
* An optional text prompt describing the desired object, material, colour, or appearance.

The application processes the sketch, prepares a suitable visual representation, reconstructs a corresponding 3D model, and allows the user to export the generated asset.

The main objective is to preserve the important characteristics of the original sketch while generating a visually meaningful 3D object.

---

## 🎯 2. Problem Statement

Converting a simple 2D drawing into a realistic 3D model is challenging because a 2D sketch contains limited information about:

* Depth and volume.
* Three-dimensional structure.
* Object proportions.
* Surface characteristics.
* Unseen parts of an object.
* Materials and textures.

Traditional modelling software requires users to manually construct these details.

Similarly, basic image-processing techniques can detect edges and contours but cannot independently reconstruct all the geometry required for a complete 3D object.

SKETCH2SHAPE AI addresses this challenge by integrating computer vision, generative AI, and image-to-3D reconstruction into one workflow.

The application aims to accept a single-view 2D sketch and generate a corresponding textured 3D model.

---

## 🎯 3. Project Objectives

The major objectives are:

1. Develop an accessible application for converting 2D sketches into 3D assets.
2. Allow users to draw sketches directly within the application.
3. Support uploading existing sketch images.
4. Process and prepare sketch images for AI-based generation.
5. Use computer vision techniques to identify important visual structures.
6. Integrate generative AI for sketch-guided visual generation where appropriate.
7. Reconstruct a corresponding three-dimensional model.
8. Preserve the recognizable structure of the original sketch.
9. Generate suitable materials, colours, and textures where supported.
10. Provide a way to preview and inspect the generated model.
11. Export the generated 3D asset in `.glb` or `.obj` format.
12. Provide a reproducible end-to-end pipeline.

---

## 💡 4. Key Features

The following features describe the intended functionality of the project. Update this list according to the features implemented in the final application.

### 4.1 Sketch Drawing Canvas

* Allows users to draw rough sketches directly.
* Supports basic drawing tools.
* Includes brush and eraser functionality.
* Provides a canvas reset option.

### 4.2 Image Upload

* Allows users to upload an existing sketch.
* Supports commonly used image formats.
* Prepares the uploaded image for further processing.

### 4.3 Text Prompt Integration

* Allows users to describe the intended object.
* Supports additional details such as colour, material, and artistic style.
* Uses the prompt as optional guidance during generation.

### 4.4 Image Preprocessing

* Image resizing.
* Background handling.
* Contrast adjustment.
* Noise reduction.
* Foreground extraction or segmentation where required.

### 4.5 AI-Based Visual Generation

* Interprets the supplied sketch.
* Identifies important contours and structural features.
* Generates or refines a visual representation for 3D reconstruction.
* Attempts to preserve the original sketch's characteristics.

### 4.6 3D Model Reconstruction

* Converts the prepared visual representation into a 3D asset.
* Estimates the object's geometry and appearance.
* Produces meaningful depth and volume.

### 4.7 3D Preview

* Allows users to inspect the generated model.
* Supports viewing the object from different angles if an interactive viewer is implemented.

### 4.8 3D Model Export

* Supports exporting the generated model.
* Intended output formats include `.glb` and `.obj`.
* Allows users to save the generated asset for further use.

---

## 🏗️ 5. System Architecture

The project follows a modular architecture consisting of multiple processing stages.

### Overall Architecture

```text
              USER
                |
                v
       2D SKETCH INPUT
                |
       +--------+--------+
       |                 |
       v                 v
  Drawing Canvas    Image Upload
       |                 |
       +--------+--------+
                |
                v
       OPTIONAL TEXT PROMPT
                |
                v
        IMAGE PREPROCESSING
                |
                v
       SKETCH UNDERSTANDING
                |
                v
     SKETCH-GUIDED IMAGE GENERATION
                |
                v
        IMAGE PREPARATION
                |
                v
       IMAGE-TO-3D RECONSTRUCTION
                |
                v
        3D MESH GENERATION
                |
                v
       MESH VALIDATION
                |
                v
        TEXTURE AND MATERI
```
=======
# sketch2shape-ai
Sketch-to-3D pipeline: Stable Diffusion + ControlNet + InstantMesh, with a Gradio app. Draw or upload a sketch, get a .glb/.obj model. Built for TechZephyr 4.0.
>>>>>>> 97d7d0d6ed42583eff20a6158388a09cc411663e
