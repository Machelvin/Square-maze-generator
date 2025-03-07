import bpy
import random
import bmesh
import math
from bpy.props import IntProperty, FloatProperty, EnumProperty, BoolProperty, StringProperty, FloatVectorProperty
from bpy_extras.io_utils import ExportHelper
import bgl
import blf
from mathutils import Vector

# Addon metadata
bl_info = {
    "name": "Square Maze Generator",
    "author": "Your Name",
    "version": (2, 28),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > {addon_name}",
    "description": "Generates a customizable 3D square maze with multiple algorithms, enhancements, and preview",
    "category": "Object",
}

# Maze Generation Functions

def generate_maze_recursive_backtracker(width, height, randomness=0.5):
    """Generate a square maze using recursive backtracking."""
    print(f"Generating square maze (Recursive Backtracking): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    stack = [(1, 1)]
    grid[1][1] = 0
    while stack:
        x, y = stack.pop()
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        if random.random() < randomness:
            random.shuffle(directions)
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width * 2 and 0 < ny < height * 2 and grid[ny][nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            next_x, next_y = random.choice(neighbors)
            grid[y + (next_y - y) // 2][x + (next_x - x) // 2] = 0
            grid[next_y][next_x] = 0
            stack.append((x, y))
            stack.append((next_x, next_y))
    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_recursive_division(width, height, randomness=0.5):
    """Generate a square maze using recursive division."""
    print(f"Generating square maze (Recursive Division): width={width}, height={height}, randomness={randomness}")
    grid = [[0] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    for y in range(0, height * 2 + 1, 2):
        for x in range(0, width * 2 + 1, 2):
            grid[y][x] = 1

    def divide(x, y, w, h):
        if w <= 2 or h <= 2:
            return
        is_horizontal = random.random() < 0.5 if w != h else w > h
        if is_horizontal:
            if h < 4:
                return
            wall_y = y + random.randrange(2, h - 1, 2)
            for x_coord in range(x, x + w):
                grid[wall_y][x_coord] = 1
            opening_x = x + random.randrange(0, w, 2)
            if random.random() < randomness and opening_x < x + w:
                grid[wall_y][opening_x] = 0
            divide(x, y, w, wall_y - y)
            divide(x, wall_y + 1, w, y + h - wall_y - 1)
        else:
            if w < 4:
                return
            wall_x = x + random.randrange(2, w - 1, 2)
            for y_coord in range(y, y + h):
                grid[y_coord][wall_x] = 1
            opening_y = y + random.randrange(0, h, 2)
            if random.random() < randomness and opening_y < y + h:
                grid[opening_y][wall_x] = 0
            divide(x, y, wall_x - x, h)
            divide(wall_x + 1, y, x + w - wall_x - 1, h)

    divide(0, 0, width * 2 + 1, height * 2 + 1)
    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_prims(width, height, randomness=0.5):
    """Generate a square maze using Prim's algorithm."""
    print(f"Generating square maze (Prim's Algorithm): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    start_x, start_y = 1, 1
    grid[start_y][start_x] = 0
    walls = []
    for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < width * 2 + 1 and 0 <= ny < height * 2 + 1:
            walls.append((nx, ny, (nx + start_x) // 2, (ny + start_y) // 2))

    while walls:
        if random.random() < randomness:
            random.shuffle(walls)
        wall = walls.pop()
        wx, wy, cx, cy = wall
        if grid[wy][wx] == 1 and (grid[cy][cx] == 0 or grid[cy][cx] == 1):
            grid[wy][wx] = 0
            grid[cy][cx] = 0
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = wx + dx, wy + dy
                if 0 <= nx < width * 2 + 1 and 0 <= ny < height * 2 + 1 and grid[ny][nx] == 1:
                    walls.append((nx, ny, (nx + wx) // 2, (ny + wy) // 2))
    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_kruskals(width, height, randomness=0.5):
    """Generate a square maze using Kruskal's algorithm."""
    print(f"Generating square maze (Kruskal's Algorithm): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    parent = {(x, y): (x, y) for y in range(height * 2 + 1) for x in range(width * 2 + 1)}

    def find(p):
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(p, q):
        parent[find(p)] = find(q)

    walls = []
    for y in range(0, height * 2 + 1, 2):
        for x in range(0, width * 2 + 1, 2):
            if x + 2 < width * 2 + 1:
                walls.append(((x, y), (x + 2, y)))
            if y + 2 < height * 2 + 1:
                walls.append(((x, y), (x, y + 2)))

    if random.random() < randomness:
        random.shuffle(walls)
    for (x1, y1), (x2, y2) in walls:
        if find((x1, y1)) != find((x2, y2)):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            grid[cy][cx] = 0
            union((x1, y1), (x2, y2))

    grid[1][0] = 0
    grid[height * 2 - 1][width * 2] = 0
    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_aldous_broder(width, height, randomness=0.5):
    """Generate a square maze using the Aldous-Broder algorithm with optimization."""
    print(f"Generating square maze (Aldous-Broder): width={width}, height={height}, randomness={randomness}")
    if width > 20 or height > 20:
        print("Warning: Aldous-Broder may be slow for grids larger than 20x20. Consider a smaller size.")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    x, y = random.randrange(1, width * 2, 2), random.randrange(1, height * 2, 2)
    grid[y][x] = 0
    unvisited = (width * height) - 1

    while unvisited > 0:
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        if random.random() < randomness:
            random.shuffle(directions)
        dx, dy = random.choice(directions)
        nx, ny = x + dx, y + dy
        if 1 <= nx < width * 2 and 1 <= ny < height * 2:
            if grid[ny][nx] == 1:
                grid[ny][nx] = 0
                grid[y + dy // 2][x + dx // 2] = 0
                unvisited -= 1
            x, y = nx, ny

    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_hunt_and_kill(width, height, randomness=0.5):
    """Generate a square maze using the Hunt-and-Kill algorithm with optimization."""
    print(f"Generating square maze (Hunt-and-Kill): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    x, y = 1, 1
    grid[y][x] = 0
    unvisited = {(x, y) for y in range(1, height * 2, 2) for x in range(1, width * 2, 2)}
    unvisited.remove((x, y))

    while unvisited:
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        if random.random() < randomness:
            random.shuffle(directions)
        unvisited_neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in unvisited:
                unvisited_neighbors.append((nx, ny))
        if unvisited_neighbors:
            nx, ny = random.choice(unvisited_neighbors)
            grid[ny][nx] = 0
            grid[y + (ny - y) // 2][x + (nx - x) // 2] = 0
            unvisited.remove((nx, ny))
            x, y = nx, ny
        else:
            found = False
            for cy in range(1, height * 2, 2):
                for cx in range(1, width * 2, 2):
                    if (cx, cy) in unvisited:
                        neighbors = []
                        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if 1 <= nx < width * 2 and 1 <= ny < height * 2 and grid[ny][nx] == 0:
                                neighbors.append((nx, ny))
                        if neighbors:
                            nx, ny = random.choice(neighbors)
                            grid[cy][cx] = 0
                            grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0
                            unvisited.remove((cx, cy))
                            x, y = cx, cy
                            found = True
                            break
                if found:
                    break
            if not found:
                break

    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_sidewinder(width, height, randomness=0.5):
    """Generate a square maze using the Sidewinder algorithm."""
    print(f"Generating square maze (Sidewinder): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    
    for y in range(1, height * 2, 2):
        run = []
        for x in range(1, width * 2, 2):
            grid[y][x] = 0
            run.append((x, y))
            if x + 2 < width * 2 and random.random() < (1 - randomness):
                grid[y][x + 1] = 0
            else:
                if run and y > 1:
                    cx, cy = random.choice(run)
                    grid[cy - 1][cx] = 0
                run = []

        if run and y > 1:
            cx, cy = random.choice(run)
            grid[cy - 1][cx] = 0

    grid[1][1] = 0
    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_growing_tree(width, height, randomness=0.5):
    """Generate a square maze using the Growing Tree algorithm."""
    print(f"Generating square maze (Growing Tree): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    active = [(1, 1)]
    grid[1][1] = 0

    while active:
        if random.random() < randomness:
            current = random.choice(active)
        else:
            current = active[-1]
        x, y = current
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        if random.random() < randomness:
            random.shuffle(directions)
        unvisited_neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width * 2 and 1 <= ny < height * 2 and grid[ny][nx] == 1:
                unvisited_neighbors.append((nx, ny))
        if unvisited_neighbors:
            next_x, next_y = random.choice(unvisited_neighbors)
            grid[next_y][next_x] = 0
            grid[y + (next_y - y) // 2][x + (next_x - x) // 2] = 0
            active.append((next_x, next_y))
        else:
            active.remove((x, y))

    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

def generate_maze_ellers(width, height, randomness=0.5):
    """Generate a square maze using Eller's algorithm."""
    print(f"Generating square maze (Eller’s): width={width}, height={height}, randomness={randomness}")
    grid = [[1] * (width * 2 + 1) for _ in range(height * 2 + 1)]
    sets = list(range(0, width * 2 + 1, 2))

    for y in range(1, height * 2, 2):
        for x in range(1, width * 2, 2):
            grid[y][x] = 0
            if x + 2 < width * 2 and random.random() < randomness:
                if sets[x // 2] != sets[(x + 2) // 2]:
                    grid[y][x + 1] = 0
                    old_set = sets[(x + 2) // 2]
                    new_set = sets[x // 2]
                    for i in range(len(sets)):
                        if sets[i] == old_set:
                            sets[i] = new_set

        if y < height * 2 - 1:
            used_sets = set()
            for x in range(1, width * 2, 2):
                if x + 2 >= width * 2 or random.random() < (1 - randomness):
                    if sets[x // 2] not in used_sets and random.random() < 0.5:
                        grid[y + 1][x] = 0
                        used_sets.add(sets[x // 2])
            if y == height * 2 - 1:
                for x in range(1, width * 2, 2):
                    if sets[x // 2] not in used_sets:
                        grid[y + 1][x] = 0

    print(f"Square grid generated: {len(grid)}x{len(grid[0])}")
    return grid

# Pathfinding Function
def find_shortest_path(grid, width, height):
    """Find the shortest path from (1, 0) to (width * 2 - 1, height * 2 - 1) using A*."""
    start = (1, 0)
    end = (width * 2 - 1, height * 2 - 1)
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}

    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        open_set.remove(current)
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < len(grid[0]) and 0 <= next_pos[1] < len(grid) and
                grid[next_pos[1]][next_pos[0]] == 0):
                neighbor = (current[0] + dx // 2, current[1] + dy // 2) if abs(dx) + abs(dy) == 2 else next_pos
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in open_set:
                        open_set.add(neighbor)
    return []

# 3D Mesh Generation with Animation
def create_maze_mesh(grid, wall_height, cell_size, wall_thickness, height_variation, add_floor, add_ceiling,
                     gradient_type, start_color, end_color, levels=1, use_texture=False, use_emissive=False,
                     path=None, animate=False):
    """Create a 3D multi-level maze with animation support."""
    print(f"Starting mesh creation with levels={levels}, use_texture={use_texture}, use_emissive={use_emissive}, animate={animate}")
    bm = bmesh.new()
    half_thickness = wall_thickness / 2
    existing_faces = set()

    def face_exists(verts):
        coords = tuple((round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) for v in verts)
        return coords in existing_faces

    def add_face(verts, frame=None):
        if not face_exists(verts):
            try:
                face = bm.faces.new(verts)
                coords = tuple((round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)) for v in verts)
                existing_faces.add(coords)
                if animate and face:
                    # Animate face visibility
                    face.hide_set(True)
                    bpy.context.scene.frame_set(frame or 1)
                    face.hide_set(False)
                    face.keyframe_insert(data_path="hide", frame=frame or 1)
                    # Add scale animation for highlight effect
                    for vert in face.verts:
                        vert.co[2] *= 1.2  # Scale up slightly
                        vert.keyframe_insert(data_path="co", frame=frame or 1)
                        vert.co[2] /= 1.2  # Scale back
                        vert.keyframe_insert(data_path="co", frame=frame + 2 if frame else 3)
                return face
            except Exception as e:
                print(f"Failed to add face with vertices {verts}: {e}")
        return None

    width = (len(grid[0]) - 1) // 2
    height = (len(grid) - 1) // 2
    verts = []
    for level in range(levels):
        level_verts = []
        for y in range(height * 2 + 1):
            row_bottom = []
            row_top = []
            for x in range(width * 2 + 1):
                h_var = random.uniform(-height_variation, height_variation) if height_variation > 0 else 0
                z_offset = level * (wall_height + 0.1)
                bottom = bm.verts.new((x * cell_size, y * cell_size, z_offset))
                top = bm.verts.new((x * cell_size, y * cell_size, z_offset + max(wall_height + h_var, 0.1)))
                row_bottom.append(bottom)
                row_top.append(top)
            level_verts.append((row_bottom, row_top))
        verts.append(level_verts)

    # Calculate total frames based on maze size
    total_frames = max(50, width * height * 2) if animate else 1
    frame_step = total_frames // (width * height * levels) if animate else 1
    current_frame = 1

    # Walls with thickness and vertical connections
    wall_faces = []
    for level in range(levels):
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == 1:
                    if y % 2 == 0 and x % 2 == 1:  # Horizontal wall
                        v0, v1, v2, v3 = (verts[level][y][0][x-1], verts[level][y][0][x+1],
                                          verts[level][y][1][x+1], verts[level][y][1][x-1])
                        v4 = bm.verts.new(((x-1) * cell_size, y * cell_size - half_thickness, v0.co.z))
                        v5 = bm.verts.new(((x+1) * cell_size, y * cell_size - half_thickness, v0.co.z))
                        v6 = bm.verts.new(((x+1) * cell_size, y * cell_size - half_thickness, v2.co.z))
                        v7 = bm.verts.new(((x-1) * cell_size, y * cell_size - half_thickness, v3.co.z))
                        faces = [
                            add_face([v0, v1, v2, v3], current_frame),
                            add_face([v4, v5, v6, v7], current_frame),
                            add_face([v0, v1, v5, v4], current_frame),
                            add_face([v1, v2, v6, v5], current_frame),
                            add_face([v2, v3, v7, v6], current_frame),
                            add_face([v3, v0, v4, v7], current_frame)
                        ]
                        wall_faces.extend(faces)
                        current_frame += frame_step
                    elif x % 2 == 0 and y % 2 == 1:  # Vertical wall
                        v0, v1, v2, v3 = (verts[level][y-1][0][x], verts[level][y+1][0][x],
                                          verts[level][y+1][1][x], verts[level][y-1][1][x])
                        v4 = bm.verts.new((x * cell_size - half_thickness, (y-1) * cell_size, v0.co.z))
                        v5 = bm.verts.new((x * cell_size - half_thickness, (y+1) * cell_size, v0.co.z))
                        v6 = bm.verts.new((x * cell_size - half_thickness, (y+1) * cell_size, v2.co.z))
                        v7 = bm.verts.new((x * cell_size - half_thickness, (y-1) * cell_size, v3.co.z))
                        faces = [
                            add_face([v0, v1, v2, v3], current_frame),
                            add_face([v4, v5, v6, v7], current_frame),
                            add_face([v0, v1, v5, v4], current_frame),
                            add_face([v1, v2, v6, v5], current_frame),
                            add_face([v2, v3, v7, v6], current_frame),
                            add_face([v3, v0, v4, v7], current_frame)
                        ]
                        wall_faces.extend(faces)
                        current_frame += frame_step
        # Vertical connections between levels
        if level < levels - 1:
            for y in range(1, height * 2, 2):
                for x in range(1, width * 2, 2):
                    if grid[y][x] == 0 and random.random() < 0.1:
                        v0 = verts[level][y][1][x]
                        v1 = verts[level + 1][y][0][x]
                        v2 = bm.verts.new((v0.co.x, v0.co.y, v0.co.z - half_thickness))
                        v3 = bm.verts.new((v1.co.x, v1.co.y, v1.co.z + half_thickness))
                        add_face([v0, v2, v3, v1], current_frame)
                        current_frame += frame_step

    # Floor and Ceiling
    for level in range(levels):
        if add_floor:
            for y in range(1, height * 2, 2):
                for x in range(1, width * 2, 2):
                    if grid[y][x] == 0:
                        v0, v1, v2, v3 = (verts[level][y-1][0][x-1], verts[level][y-1][0][x+1],
                                          verts[level][y+1][0][x+1], verts[level][y+1][0][x-1])
                        add_face([v0, v1, v2, v3], current_frame if animate else None)
        if add_ceiling:
            for y in range(1, height * 2, 2):
                for x in range(1, width * 2, 2):
                    if grid[y][x] == 0:
                        v0, v1, v2, v3 = (verts[level][y-1][1][x-1], verts[level][y-1][1][x+1],
                                          verts[level][y+1][1][x+1], verts[level][y+1][1][x-1])
                        add_face([v0, v1, v2, v3], current_frame if animate else None)

    # Path Visualization
    if path:
        path_verts = []
        for level in range(levels):
            z_offset = level * (wall_height + 0.1)
            for px, py in path:
                v = bm.verts.new((px * cell_size, py * cell_size, z_offset + wall_height / 2))
                path_verts.append(v)
        for i in range(0, len(path_verts) - levels, levels):
            for lvl in range(levels):
                idx = i + lvl
                if idx + levels < len(path_verts):
                    bm.edges.new([path_verts[idx], path_verts[idx + levels]])

    print(f"Total faces created: {len(bm.faces)}")
    if len(bm.faces) == 0:
        print("Error: No faces were created in the mesh.")
        bm.free()
        return None

    # Material Setup
    mat = bpy.data.materials.new(name="MazeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    vertex_color = nodes.new('ShaderNodeVertexColor')
    output = nodes.new('ShaderNodeOutputMaterial')

    if use_texture:
        tex_image = nodes.new('ShaderNodeTexBrick')
        tex_image.inputs['Scale'].default_value = 5.0
        links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
        links.new(vertex_color.outputs['Color'], tex_image.inputs['Mortar Color'])
    else:
        links.new(vertex_color.outputs['Color'], principled.inputs['Base Color'])

    if use_emissive:
        emission = nodes.new('ShaderNodeEmission')
        links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
        mix_shader = nodes.new('ShaderNodeMixShader')
        links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
        links.new(emission.outputs['Emission'], mix_shader.inputs[2])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])
        mix_shader.inputs['Fac'].default_value = 0.3
    else:
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Create mesh and assign material
    mesh = bpy.data.meshes.new("MazeMesh")
    bm.to_mesh(mesh)
    bm.free()

    mesh.uv_layers.new(name="UVMap")
    color_layer = mesh.vertex_colors.new(name="Gradient")

    center_x = width * cell_size
    center_y = height * cell_size
    max_distance = math.sqrt((center_x) ** 2 + (center_y) ** 2)
    max_x = width * 2 * cell_size
    max_y = height * 2 * cell_size

    for vert in mesh.vertices:
        if gradient_type == "radial":
            distance = math.sqrt((vert.co.x - center_x) ** 2 + (vert.co.y - center_y) ** 2)
            normalized_value = min(distance / max_distance, 1.0)
        elif gradient_type == "horizontal":
            normalized_value = vert.co.x / max_x
        elif gradient_type == "vertical":
            normalized_value = vert.co.y / max_y

        normalized_value = max(0.0, min(normalized_value, 1.0))
        if vert.index < 5:
            print(f"Vertex {vert.index} at ({vert.co.x}, {vert.co.y}, {vert.co.z}): normalized_value={normalized_value}")

        r = start_color[0] + (end_color[0] - start_color[0]) * normalized_value
        g = start_color[1] + (end_color[1] - start_color[1]) * normalized_value
        b = start_color[2] + (end_color[2] - start_color[2]) * normalized_value

        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                if mesh.loops[loop_index].vertex_index == vert.index:
                    color_layer.data[loop_index].color = (r, g, b, 1.0)

    # Path Material
    if path:
        path_mat = bpy.data.materials.new(name="PathMaterial")
        path_mat.use_nodes = True
        path_nodes = path_mat.node_tree.nodes
        path_links = path_mat.node_tree.links
        path_nodes.clear()
        path_principled = path_nodes.new('ShaderNodeBsdfPrincipled')
        path_principled.inputs['Base Color'].default_value = (0.0, 1.0, 0.0, 1.0)  # Green path
        path_output = path_nodes.new('ShaderNodeOutputMaterial')
        path_links.new(path_principled.outputs['BSDF'], path_output.inputs['Surface'])
        mesh.materials.append(mat)
        mesh.materials.append(path_mat)
        # Assign path material to path edges
        for edge in mesh.edges:
            edge.use_freestyle_mark = True
        bpy.context.view_layer.objects.active = bpy.context.scene.objects.get("Maze")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for edge in mesh.edges:
            if edge.use_freestyle_mark:
                edge.select = True
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.object.material_slot_assign()

    print("Mesh created successfully with enhancements")
    return mesh

# Preview Drawing
preview_handle = None
preview_grid = None

def draw_preview():
    global preview_grid
    if not preview_grid:
        return
    width, height = len(preview_grid[0]) // 2, len(preview_grid) // 2
    region = bpy.context.region
    scale = min(region.width, region.height) / max(width, height) * 0.8
    offset_x = region.width / 2 - (width * scale) / 2
    offset_y = region.height / 2 - (height * scale) / 2

    bgl.glEnable(bgl.GL_BLEND)
    bgl.glColor4f(1.0, 1.0, 1.0, 0.8)

    for y in range(len(preview_grid)):
        for x in range(len(preview_grid[0])):
            if preview_grid[y][x] == 1:
                bgl.glBegin(bgl.GL_QUADS)
                bgl.glVertex2f(offset_x + x * scale, offset_y + y * scale)
                bgl.glVertex2f(offset_x + (x + 1) * scale, offset_y + y * scale)
                bgl.glVertex2f(offset_x + (x + 1) * scale, offset_y + (y + 1) * scale)
                bgl.glVertex2f(offset_x + x * scale, offset_y + (y + 1) * scale)
                bgl.glEnd()

    bgl.glDisable(bgl.GL_BLEND)
    blf.position(0, 30, 0)
    blf.size(0, 14, 72)
    blf.draw(0, f"Preview: {width}x{height} Maze")

def update_preview(self, context):
    global preview_handle, preview_grid
    if preview_handle:
        bpy.types.SpaceView3D.draw_handler_remove(preview_handle, 'WINDOW')
    scene = context.scene
    if scene.maze_algorithm and scene.maze_width > 0 and scene.maze_height > 0:
        alg_func = {
            "recursive_backtracking": generate_maze_recursive_backtracker,
            "recursive_division": generate_maze_recursive_division,
            "prims": generate_maze_prims,
            "kruskals": generate_maze_kruskals,
            "aldous_broder": generate_maze_aldous_broder,
            "hunt_and_kill": generate_maze_hunt_and_kill,
            "sidewinder": generate_maze_sidewinder,
            "growing_tree": generate_maze_growing_tree,
            "ellers": generate_maze_ellers
        }[scene.maze_algorithm]
        preview_grid = alg_func(scene.maze_width, scene.maze_height, scene.maze_randomness)
        preview_handle = bpy.types.SpaceView3D.draw_handler_add(draw_preview, (), 'WINDOW', 'POST_PIXEL')
    else:
        preview_grid = None

# Operator for Maze Generation
class MazeGeneratorOperator(bpy.types.Operator):
    """Operator to generate a 3D square maze in Blender."""
    bl_idname = "object.maze_generator"
    bl_label = "Generate Maze"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global preview_handle, preview_grid
        scene = context.scene
        print("Starting maze generation operator...")

        if preview_handle:
            bpy.types.SpaceView3D.draw_handler_remove(preview_handle, 'WINDOW')
            preview_handle = None
            preview_grid = None

        alg_func = {
            "recursive_backtracking": generate_maze_recursive_backtracker,
            "recursive_division": generate_maze_recursive_division,
            "prims": generate_maze_prims,
            "kruskals": generate_maze_kruskals,
            "aldous_broder": generate_maze_aldous_broder,
            "hunt_and_kill": generate_maze_hunt_and_kill,
            "sidewinder": generate_maze_sidewinder,
            "growing_tree": generate_maze_growing_tree,
            "ellers": generate_maze_ellers
        }[scene.maze_algorithm]
        grid = alg_func(scene.maze_width, scene.maze_height, scene.maze_randomness)

        if not grid:
            self.report({'ERROR'}, "Failed to generate maze grid.")
            return {'CANCELLED'}

        if scene.maze_add_openings:
            grid[1][0] = 0
            grid[scene.maze_height * 2 - 1][scene.maze_width * 2] = 0

        path = find_shortest_path(grid, scene.maze_width, scene.maze_height) if scene.maze_show_path else None
        mesh = create_maze_mesh(
            grid,
            scene.maze_wall_height,
            scene.maze_cell_size,
            scene.maze_wall_thickness,
            scene.maze_height_variation,
            scene.maze_add_floor,
            scene.maze_add_ceiling,
            scene.maze_gradient_type,
            scene.maze_gradient_start_color,
            scene.maze_gradient_end_color,
            scene.maze_levels,
            scene.maze_use_texture,
            scene.maze_use_emissive,
            path,
            scene.maze_animate
        )
        if not mesh:
            self.report({'ERROR'}, "Failed to create maze mesh.")
            return {'CANCELLED'}

        obj = bpy.data.objects.new("Maze", mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)
        print(f"Object 'Maze' created and linked to scene. Visible: {not obj.hide_viewport}")

        if scene.maze_animate:
            # Set animation range
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = total_frames  # Set in create_maze_mesh
            # Ensure playback starts
            bpy.context.scene.frame_set(1)
            bpy.ops.screen.animation_play()

        self.report({'INFO'}, "Maze generated successfully.")
        return {'FINISHED'}

# Random Settings Operator
class RandomSettingsOperator(bpy.types.Operator):
    """Generate random settings for maze creation."""
    bl_idname = "object.random_settings"
    bl_label = "Generate Random Settings"

    def execute(self, context):
        scene = context.scene
        scene.maze_width = random.randint(3, 10)
        scene.maze_height = random.randint(3, 10)
        scene.maze_randomness = random.uniform(0.0, 1.0)
        scene.maze_levels = random.randint(1, 3)
        scene.maze_use_texture = random.choice([True, False])
        scene.maze_use_emissive = random.choice([True, False])
        scene.maze_show_path = random.choice([True, False])
        scene.maze_gradient_type = random.choice(["radial", "horizontal", "vertical"])
        scene.maze_gradient_start_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        scene.maze_gradient_end_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        scene.maze_algorithm = random.choice([
            "recursive_backtracking", "recursive_division", "prims", "kruskals",
            "aldous_broder", "hunt_and_kill", "sidewinder", "growing_tree", "ellers"
        ])
        update_preview(None, context)
        self.report({'INFO'}, "Random settings applied.")
        return {'FINISHED'}

# UI Panel
class MazeGeneratorPanel(bpy.types.Panel):
    """Panel in the 3D Viewport for maze generation settings."""
    bl_label = "Square Maze Generator"
    bl_idname = "OBJECT_PT_maze_generator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "{addon_name}"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Maze Settings")
        layout.prop(scene, "maze_width")
        layout.prop(scene, "maze_height")
        layout.prop(scene, "maze_levels")
        layout.prop(scene, "maze_wall_height")
        layout.prop(scene, "maze_cell_size")
        layout.prop(scene, "maze_wall_thickness")
        layout.prop(scene, "maze_randomness")
        layout.prop(scene, "maze_height_variation")
        layout.prop(scene, "maze_add_floor")
        layout.prop(scene, "maze_add_ceiling")
        layout.prop(scene, "maze_add_openings")
        layout.prop(scene, "maze_animate")
        layout.prop(scene, "maze_show_path")
        layout.prop(scene, "maze_use_texture")
        layout.prop(scene, "maze_use_emissive")
        layout.prop(scene, "maze_algorithm")
        layout.prop(scene, "maze_gradient_type")
        layout.prop(scene, "maze_gradient_start_color", text="Start Color")
        layout.prop(scene, "maze_gradient_end_color", text="End Color")

        layout.operator("object.random_settings", text="Generate Random Settings")
        layout.operator("object.maze_generator", text="Generate Maze")
        layout.operator("object.maze_export", text="Export Maze as STL")

# Export Operator
class MazeExportOperator(bpy.types.Operator, ExportHelper):
    """Export the maze as STL."""
    bl_idname = "object.maze_export"
    bl_label = "Export Maze as STL"
    filename_ext = ".stl"

    filepath: StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        if bpy.context.active_object and bpy.context.active_object.type == 'MESH':
            bpy.ops.export_mesh.stl(filepath=self.filepath)
            self.report({'INFO'}, f"Maze exported to {self.filepath}")
        else:
            self.report({'ERROR'}, "No maze selected!")
        return {'FINISHED'}

def register_properties():
    bpy.types.Scene.maze_width = IntProperty(name="Width", default=5, min=1, max=50)
    bpy.types.Scene.maze_height = IntProperty(name="Height", default=5, min=1, max=50)
    bpy.types.Scene.maze_levels = IntProperty(name="Levels", default=1, min=1, max=5)
    bpy.types.Scene.maze_wall_height = FloatProperty(name="Wall Height", default=2.0, min=0.1)
    bpy.types.Scene.maze_cell_size = FloatProperty(name="Cell Size", default=1.0, min=0.1)
    bpy.types.Scene.maze_wall_thickness = FloatProperty(name="Wall Thickness", default=0.05, min=0.01, max=0.5)
    bpy.types.Scene.maze_randomness = FloatProperty(name="Randomness", default=0.5, min=0.0, max=1.0)
    bpy.types.Scene.maze_height_variation = FloatProperty(name="Height Variation", default=0.0, min=0.0, max=1.0)
    bpy.types.Scene.maze_add_floor = BoolProperty(name="Add Floor", default=False)
    bpy.types.Scene.maze_add_ceiling = BoolProperty(name="Add Ceiling", default=False)
    bpy.types.Scene.maze_add_openings = BoolProperty(name="Add Openings", default=False)
    bpy.types.Scene.maze_animate = BoolProperty(name="Animate Generation", default=False)
    bpy.types.Scene.maze_show_path = BoolProperty(name="Show Shortest Path", default=False)
    bpy.types.Scene.maze_use_texture = BoolProperty(name="Use Brick Texture", default=False)
    bpy.types.Scene.maze_use_emissive = BoolProperty(name="Use Emissive Gradient", default=False)
    bpy.types.Scene.maze_algorithm = EnumProperty(
        items=[
            ("recursive_backtracking", "Recursive Backtracking", "Perfect maze with a single solution"),
            ("recursive_division", "Recursive Division", "Structured maze with random openings"),
            ("prims", "Prim's Algorithm", "Random maze with potential multiple solutions"),
            ("kruskals", "Kruskal's Algorithm", "Random maze with potential multiple solutions"),
            ("aldous_broder", "Aldous-Broder", "Uniform random maze, slower but unbiased"),
            ("hunt_and_kill", "Hunt-and-Kill", "Organic maze with random walks and hunting"),
            ("sidewinder", "Sidewinder", "Maze with horizontal bias, longer east-west passages"),
            ("growing_tree", "Growing Tree", "Maze grown from active cells with customizable bias"),
            ("ellers", "Eller’s", "Row-based maze with ensured connectivity")
        ],
        name="Algorithm",
        default="recursive_backtracking",
        update=update_preview
    )
    bpy.types.Scene.maze_gradient_type = EnumProperty(
        items=[
            ("radial", "Radial", "Gradient from center to edge"),
            ("horizontal", "Horizontal", "Gradient from left to right"),
            ("vertical", "Vertical", "Gradient from bottom to top")
        ],
        name="Gradient Type",
        default="radial",
    )
    bpy.types.Scene.maze_gradient_start_color = FloatVectorProperty(
        name="Gradient Start Color",
        subtype='COLOR',
        default=(0.0, 0.0, 1.0),  # Blue
        min=0.0, max=1.0
    )
    bpy.types.Scene.maze_gradient_end_color = FloatVectorProperty(
        name="Gradient End Color",
        subtype='COLOR',
        default=(1.0, 0.0, 0.0),  # Red
        min=0.0, max=1.0
    )

def unregister_properties():
    del bpy.types.Scene.maze_width
    del bpy.types.Scene.maze_height
    del bpy.types.Scene.maze_levels
    del bpy.types.Scene.maze_wall_height
    del bpy.types.Scene.maze_cell_size
    del bpy.types.Scene.maze_wall_thickness
    del bpy.types.Scene.maze_randomness
    del bpy.types.Scene.maze_height_variation
    del bpy.types.Scene.maze_add_floor
    del bpy.types.Scene.maze_add_ceiling
    del bpy.types.Scene.maze_add_openings
    del bpy.types.Scene.maze_animate
    del bpy.types.Scene.maze_show_path
    del bpy.types.Scene.maze_use_texture
    del bpy.types.Scene.maze_use_emissive
    del bpy.types.Scene.maze_algorithm
    del bpy.types.Scene.maze_gradient_type
    del bpy.types.Scene.maze_gradient_start_color
    del bpy.types.Scene.maze_gradient_end_color

def register():
    current_text = bpy.context.space_data.text
    addon_name = current_text.name if current_text else "Tools"
    bl_info['location'] = bl_info['location'].format(addon_name=addon_name)
    MazeGeneratorPanel.bl_category = addon_name
    register_properties()
    bpy.utils.register_class(MazeGeneratorOperator)
    bpy.utils.register_class(RandomSettingsOperator)
    bpy.utils.register_class(MazeExportOperator)
    bpy.utils.register_class(MazeGeneratorPanel)

def unregister():
    global preview_handle
    if preview_handle:
        bpy.types.SpaceView3D.draw_handler_remove(preview_handle, 'WINDOW')
    unregister_properties()
    bpy.utils.unregister_class(MazeGeneratorOperator)
    bpy.utils.unregister_class(RandomSettingsOperator)
    bpy.utils.unregister_class(MazeExportOperator)
    bpy.utils.unregister_class(MazeGeneratorPanel)

if __name__ == "__main__":
    register()