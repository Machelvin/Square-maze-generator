Square Maze Generator Addon Documentation
Overview
The Square Maze Generator is a Blender addon designed to create customizable 3D square mazes within Blender 4.3.0. It supports multiple maze generation algorithms, multi-level mazes, pathfinding visualization, material customization with gradients and textures, and export functionality. The addon provides an intuitive interface in the 3D Viewport sidebar and includes a live preview of the maze layout.

Addon Metadata
Name: Square Maze Generator
Author: Kelvin Macharia, Grok (xAI)
Version: 1.0
Blender Version: 4.3.0
Location: View3D > Sidebar > Square Maze Generator
Description: Generates a customizable 3D square maze with multiple algorithms, enhancements, and preview.
Category: Object
Installation
Download the Script:
Save the script as square_maze_generator.py on your computer.
Install in Blender:
Open Blender 4.3.0.
Go to Edit > Preferences > Add-ons.
Click Install... and select the square_maze_generator.py file.
Enable the addon by checking the box next to "Object: Square Maze Generator".
Access the Addon:
In the 3D Viewport, press N to open the sidebar.
Navigate to the "Square Maze Generator" tab to see the addon’s panel.
User Guide
Interface
The addon’s panel is located in the 3D Viewport sidebar under the "Square Maze Generator" tab. It contains settings for maze generation, appearance customization, and actions.

Maze Settings
Width: Number of cells along the X-axis (1 to 50, default: 5).
Height: Number of cells along the Y-axis (1 to 50, default: 5).
Levels: Number of vertical maze levels (1 to 5, default: 1).
Wall Height: Height of the maze walls (minimum 0.1, default: 2.0).
Cell Size: Size of each maze cell (minimum 0.1, default: 1.0).
Wall Thickness: Thickness of the maze walls (0.01 to 0.5, default: 0.05).
Randomness: Controls randomness in maze generation (0.0 to 1.0, default: 0.5).
Height Variation: Random variation in wall height (0.0 to 1.0, default: 0.0).
Add Floor: Toggle to add a floor to the maze (default: False).
Add Ceiling: Toggle to add a ceiling to the maze (default: False).
Add Openings: Toggle to add entrance and exit openings (default: False).
Show Shortest Path: Toggle to visualize the shortest path from top-left to bottom-right (default: False).
Use Brick Texture: Toggle to apply a brick texture to the maze (default: False).
Use Emissive Gradient: Toggle to add an emissive effect to the gradient (default: False).
Algorithm: Select the maze generation algorithm (default: Recursive Backtracking).
Options:
Recursive Backtracking: Perfect maze with a single solution.
Recursive Division: Structured maze with random openings.
Prim’s Algorithm: Random maze with potential multiple solutions.
Kruskal’s Algorithm: Random maze with potential multiple solutions.
Aldous-Broder: Uniform random maze, slower but unbiased.
Hunt-and-Kill: Organic maze with random walks and hunting.
Sidewinder: Maze with horizontal bias, longer east-west passages.
Growing Tree: Maze grown from active cells with customizable bias.
Eller’s: Row-based maze with ensured connectivity.
Gradient Type: Select the gradient style (default: Radial).
Options: Radial, Horizontal, Vertical.
Start Color: Starting color for the gradient (default: Blue, RGB: 0, 0, 1).
End Color: Ending color for the gradient (default: Red, RGB: 1, 0, 0).
Actions
Generate Random Settings: Randomizes maze parameters for quick experimentation.
Generate Maze: Creates the 3D maze based on current settings.
Export Maze as STL: Exports the generated maze as an STL file.
Usage
Adjust Settings:
Modify the maze dimensions, appearance, and algorithm as desired.
Use the live preview to see the 2D maze layout as you change settings (e.g., width, height, algorithm).
Generate the Maze:
Click the "Generate Maze" button to create the 3D maze.
The maze will appear in the 3D Viewport as a mesh object named "Maze".
Visualize the Path:
If "Show Shortest Path" is enabled, a green line will trace the shortest path from the top-left to the bottom-right of the maze.
Export the Maze:
After generating the maze, click "Export Maze as STL" to save the mesh as an STL file for 3D printing or other uses.
Technical Documentation
Dependencies
The addon relies on the following Blender modules and Python libraries:

bpy: Blender Python API for scene manipulation.
random: For random number generation and shuffling.
bmesh: For creating and manipulating mesh data.
math: For mathematical calculations.
bpy.props: For defining custom properties.
bpy_extras.io_utils.ExportHelper: For file export functionality.
bgl: For OpenGL drawing in the viewport (used in preview).
blf: For font rendering in the viewport (used in preview).
mathutils.Vector: For vector operations.
Maze Generation
The addon implements nine maze generation algorithms, each producing a grid where 0 represents a path and 1 represents a wall:

Recursive Backtracking: Uses a stack-based depth-first search to carve paths.
Recursive Division: Divides the grid recursively, adding walls with random openings.
Prim’s Algorithm: Grows the maze from a starting point by randomly connecting cells.
Kruskal’s Algorithm: Uses a union-find structure to connect cells while avoiding cycles.
Aldous-Broder: Randomly walks through the grid, carving paths until all cells are visited.
Hunt-and-Kill: Combines random walks with a hunting phase to connect unvisited cells.
Sidewinder: Generates mazes row by row with a horizontal bias.
Growing Tree: Grows the maze by selecting active cells with a configurable bias.
Eller’s Algorithm: Builds the maze row by row, ensuring connectivity.
Each algorithm takes width, height, and randomness parameters and returns a 2D grid of size (height * 2 + 1) x (width * 2 + 1).

Mesh Creation
The create_maze_mesh function converts the 2D grid into a 3D mesh using bmesh:

Grid Structure: The grid is interpreted as cells (paths) and walls. Each cell is 2x2 units in the grid, with walls between them.
Vertices and Faces: Creates vertices for each grid point at bottom and top heights, adjusted by cell_size and wall_height. Walls are extruded with thickness using wall_thickness.
Multi-Level: Stacks mazes vertically if levels > 1, with random vertical connections between levels.
Floor and Ceiling: Adds floor and ceiling faces if enabled.
Path Visualization: If a path is provided, draws it as a series of edges with a distinct material.
Materials:
Applies a gradient based on gradient_type (Radial, Horizontal, Vertical) using vertex colors.
Supports a brick texture (use_texture) and emissive effects (use_emissive).
Pathfinding
The find_shortest_path function uses Breadth-First Search (BFS) to find the shortest path from the top-left (1, 1) to the bottom-right (width * 2 - 1, height * 2 - 1) of the maze grid. The path is visualized as a green line if maze_show_path is enabled.

Live Preview
Functionality: The draw_preview function uses bgl to draw a 2D representation of the maze in the viewport.
Update: The update_preview function regenerates the preview grid whenever maze_width, maze_height, or maze_algorithm changes.
Display: Walls are drawn as white quads with 80% opacity, and a text label shows the maze dimensions.
Operators
MazeGeneratorOperator (object.maze_generator): Generates the 3D maze mesh.
RandomSettingsOperator (object.random_settings): Randomizes maze settings.
MazeExportOperator (object.maze_export): Exports the maze as an STL file.
Properties
Custom properties are defined on bpy.types.Scene for user input, including integers, floats, booleans, enums, and color vectors.

Known Limitations
Performance: Algorithms like Aldous-Broder can be slow for large grids (e.g., >20x20).
Mesh Complexity: High values for width, height, or levels can result in complex meshes, potentially slowing down Blender.
Preview: The live preview may not scale well for very large mazes due to viewport resolution.
Troubleshooting
No Panel in Sidebar:
Ensure the addon is enabled in Edit > Preferences > Add-ons.
Check that you’re in the 3D Viewport and press N to open the sidebar.
Maze Not Generating:
Verify that width and height are at least 1.
Check the Blender System Console for error messages.
Export Fails:
Ensure a maze has been generated and is selected before exporting.
Slow Performance:
Reduce width, height, or levels.
Avoid using slow algorithms like Aldous-Broder for large mazes.
Developer Notes
Extending Algorithms: Add new algorithms by defining a function generate_maze_<name> and updating the alg_func dictionary in MazeGeneratorOperator and update_preview.
Custom Materials: Modify the create_maze_mesh material setup to support additional textures or shaders.
Improving Preview: Enhance draw_preview to support more detailed visualizations, such as showing the path or multi-level connections.
Changes Made
Updated the version in the bl_info dictionary from (2, 29) to (1, 0) in the script.
Updated the Version field in the documentation’s Overview section from "2.29" to "1.0".
Kept the author as "Kelvin Macharia, Grok (xAI)" as per your previous request.
Notes
The version (1, 0) is interpreted as "1.0" in the documentation for readability, following Blender’s convention of displaying tuple versions as dotted strings (e.g., (1, 0) becomes "1.0").
If you meant a different version format (e.g., (1,) or a single integer 1), please let me know, and I’ll adjust accordingly.
Next Steps
Replace the bl_info section in your script with the updated one above.
The documentation is ready to use. If you’d like further adjustments (e.g., a different version number, additional sections, or a different format like Markdown), just let me know
