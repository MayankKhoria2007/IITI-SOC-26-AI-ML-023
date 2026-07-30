# Interior AI Studio

**Intelligent Interior Design Platform — see the change before you make it.**

Interior design decisions are expensive to get wrong. Picking the wrong paint colour, furniture piece, or layout usually means rework, wasted money, and a room that still doesn't look right. Professional design consultation is out of reach for most people, and a mood board on paper doesn't show how a change will actually look in someone's real space.

Interior AI Studio solves this with **virtual try-on for interiors** — upload a photo (or start from nothing), describe or drag what you want to change, and see a photorealistic result instantly, with a full branching history of every variation you try.

---

## Core Modules

Interior AI Studio is built around five modules that together cover the full lifecycle of a design idea, from a blank page to a finished, explorable room.

### A — Image-Based Interior Editing
Upload a photo of your room and describe the change in plain language — *"change the bedsheet and pillows to navy blue"* — and the model applies just that edit while leaving the rest of the room untouched. This is the flagship module, powered by our optimised **FLUX.1-Kontext** pipeline.

### B — Text-to-Interior Generation
No existing photo? Describe the room you want and generate a photorealistic interior concept from text alone. The result feeds directly into Module A for further refinement.

### C — Furniture Inpainting
Provide an interior photo and a photo of a furniture piece you want to place in it. The module detects and masks the existing item, then composites the new piece in place — matching perspective, lighting, scale, and shadow.

### D — Drag-and-Drop Object Repositioning
Select an object in the room, and the module segments and extracts it, inpainting the vacated area behind it. Drag the object to a new spot, and a depth-analysis pass automatically rescales it to match its new position in the scene (closer = larger, farther = smaller), keeping perspective and shadows consistent.

### E — 3D View Synthesis
Using the same depth-analysis pipeline as Module D, this module reconstructs an approximate 3D representation of the room from a single 2D photo, letting you explore the space from angles beyond the original shot.

| Module | Input | Output |
|---|---|---|
| A. Image Editing | Room photo + text prompt | Edited room photo |
| B. Text-to-Interior | Text description | Generated room photo |
| C. Furniture Inpainting | Room photo + furniture photo | Room with furniture composited in |
| D. Object Repositioning | Room photo + drag target | Room with object moved & rescaled |
| E. 3D View Synthesis | Room photo | Explorable 3D-approximate view |

---

## Web Application

The web application (frontend + backend implementation, architecture docs, and setup instructions) lives in its own repository:

🔗 **[IITISOC-InteriorAI-Studio](https://github.com/Prathamesh-Hingol/IITISOC-InteriorAI-Studio)**

---


