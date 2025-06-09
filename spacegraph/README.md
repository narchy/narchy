# SpaceGraph - fractal computer interface

A declarative UI framework for building dynamic 2D interfaces with physics-inspired layouts and touch-first interaction. Combines Java2D-like composition with OpenGL acceleration.

## Core Architecture

### Surface Hierarchy
- **Surface**: Base abstract class for all renderables
  - Manages position/bounds, visibility, parenting
  - Implements hierarchical scene graph traversal
  - Handles focus and Z-ordering
- **Surfacelike**: Interface for scene graph participants
- **ContainerSurface**: Abstract container with layout logic

### Layout System
- **Bordering**: Border-style layouts (N/S/E/W/Center)
- **ScrollXY**: Scrollable viewport with momentum
- **Splitting**: Resizable split panels (horizontal/vertical)
- **Springing**: Physics-inspired dynamic layouts
- **RingContainer**: Circular/radial layouts

### Input System
- **Finger**: Touch/pointer state tracking
  - Position normalization across surfaces
  - Button state management (press/release/drag)
- **Fingering**: Gesture handling base class
  - Custom gesture implementation via start/update/stop
- **FingerRenderer**: Visual feedback for interactions

### Rendering Pipeline
1. **JoglWindow**: GL context manager
   - VSync-controlled render loop
   - Multi-layer compositing
2. **Layer**: Render stage interface
   - Handles GL initialization
   - Manages render state changes
3. **ReSurface**: Render context propagator
   - Maintains coordinate transforms
   - Manages visibility culling

## Key Features
- Declarative UI composition
- Physics-inspired layout managers
- Touch-optimized interaction model
- GL accelerated rendering
- Cross-platform (JOGL-based)
- Lightweight dependency profile
