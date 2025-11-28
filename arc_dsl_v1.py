"""
ARC Prize 2025 - DSL Implementation V1
40 Essential Primitives based on MindsAI DSL analysis

Categories:
1. Grid Transformations (10)
2. Color Operations (8)
3. Object Operations (8)
4. Fill/Paint (6)
5. Geometric (4)
6. Index/Position (2)
7. Advanced (2)
"""

import numpy as np
from typing import Tuple, FrozenSet, Set, Callable, Any, List
from collections import Counter

# Type definitions
Grid = Tuple[Tuple[int, ...], ...]
Indices = FrozenSet[Tuple[int, int]]
Object = FrozenSet[Tuple[int, Tuple[int, int]]]
Objects = FrozenSet[Object]
Patch = Indices  # Simplified for now


class ARCDSL:
    """40 Essential DSL Primitives for ARC-AGI"""

    # =====================================
    # Group 1: Grid Transformations (10)
    # =====================================

    @staticmethod
    def rot90(grid: Grid) -> Grid:
        """Quarter clockwise rotation"""
        return tuple(row for row in zip(*grid[::-1]))

    @staticmethod
    def rot180(grid: Grid) -> Grid:
        """Half rotation"""
        return tuple(tuple(row[::-1]) for row in grid[::-1])

    @staticmethod
    def rot270(grid: Grid) -> Grid:
        """Quarter anticlockwise rotation"""
        return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

    @staticmethod
    def hmirror(grid: Grid) -> Grid:
        """Horizontal mirror (flip vertically)"""
        return grid[::-1]

    @staticmethod
    def vmirror(grid: Grid) -> Grid:
        """Vertical mirror (flip horizontally)"""
        return tuple(row[::-1] for row in grid)

    @staticmethod
    def dmirror(grid: Grid) -> Grid:
        """Diagonal mirror (transpose)"""
        return tuple(zip(*grid))

    @staticmethod
    def upscale(grid: Grid, factor: int) -> Grid:
        """Upscale grid by factor"""
        result = []
        for row in grid:
            upscaled_row = []
            for value in row:
                upscaled_row.extend([value] * factor)
            for _ in range(factor):
                result.append(tuple(upscaled_row))
        return tuple(result)

    @staticmethod
    def downscale(grid: Grid, factor: int) -> Grid:
        """Downscale grid by factor"""
        h, w = len(grid), len(grid[0])
        result = []
        for i in range(0, h, factor):
            if i < h:
                row = []
                for j in range(0, w, factor):
                    if j < w:
                        row.append(grid[i][j])
                if row:
                    result.append(tuple(row))
        return tuple(result)

    @staticmethod
    def hconcat(a: Grid, b: Grid) -> Grid:
        """Concatenate two grids horizontally"""
        return tuple(row_a + row_b for row_a, row_b in zip(a, b))

    @staticmethod
    def vconcat(a: Grid, b: Grid) -> Grid:
        """Concatenate two grids vertically"""
        return a + b

    # =====================================
    # Group 2: Color Operations (8)
    # =====================================

    @staticmethod
    def palette(grid: Grid) -> FrozenSet[int]:
        """Get all colors in grid"""
        colors = set()
        for row in grid:
            colors.update(row)
        return frozenset(colors)

    @staticmethod
    def mostcolor(grid: Grid) -> int:
        """Most common color in grid"""
        values = [v for row in grid for v in row]
        if not values:
            return 0
        return max(set(values), key=values.count)

    @staticmethod
    def leastcolor(grid: Grid) -> int:
        """Least common color in grid (excluding 0 if possible)"""
        values = [v for row in grid for v in row]
        if not values:
            return 0
        color_counts = Counter(values)
        # Try to avoid returning 0 (background)
        non_zero = {c: cnt for c, cnt in color_counts.items() if c != 0}
        if non_zero:
            return min(non_zero, key=non_zero.get)
        return min(color_counts, key=color_counts.get)

    @staticmethod
    def colorfilter(objects: Objects, value: int) -> Objects:
        """Filter objects by color"""
        result = set()
        for obj in objects:
            if obj and next(iter(obj))[0] == value:
                result.add(obj)
        return frozenset(result)

    @staticmethod
    def colorcount(grid: Grid, value: int) -> int:
        """Count cells with specific color"""
        count = 0
        for row in grid:
            count += row.count(value)
        return count

    @staticmethod
    def replace(grid: Grid, old_color: int, new_color: int) -> Grid:
        """Replace color in grid"""
        return tuple(
            tuple(new_color if v == old_color else v for v in row)
            for row in grid
        )

    @staticmethod
    def switch(grid: Grid, color1: int, color2: int) -> Grid:
        """Switch two colors"""
        return tuple(
            tuple(
                color2 if v == color1 else (color1 if v == color2 else v)
                for v in row
            )
            for row in grid
        )

    @staticmethod
    def recolor(obj: Object, new_color: int) -> Object:
        """Recolor an object"""
        return frozenset((new_color, loc) for _, loc in obj)

    # =====================================
    # Group 3: Object Operations (8)
    # =====================================

    @staticmethod
    def objects(grid: Grid, diagonal: bool = True, without_bg: bool = True) -> Objects:
        """Extract objects from grid"""
        h, w = len(grid), len(grid[0])
        bg = ARCDSL.mostcolor(grid) if without_bg else None
        objs = set()
        occupied = set()

        for i in range(h):
            for j in range(w):
                if (i, j) in occupied:
                    continue
                val = grid[i][j]
                if val == bg:
                    continue

                # BFS to find connected component
                obj = set()
                queue = [(i, j)]
                while queue:
                    ci, cj = queue.pop(0)
                    if (ci, cj) in occupied:
                        continue
                    if not (0 <= ci < h and 0 <= cj < w):
                        continue
                    if grid[ci][cj] != val:
                        continue

                    obj.add((val, (ci, cj)))
                    occupied.add((ci, cj))

                    # Add neighbors
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        queue.append((ci+di, cj+dj))
                    if diagonal:
                        for di, dj in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            queue.append((ci+di, cj+dj))

                if obj:
                    objs.add(frozenset(obj))

        return frozenset(objs)

    @staticmethod
    def partition(grid: Grid) -> Objects:
        """Partition grid by color"""
        colors = ARCDSL.palette(grid)
        result = set()
        h, w = len(grid), len(grid[0])

        for color in colors:
            obj = set()
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == color:
                        obj.add((color, (i, j)))
            if obj:
                result.add(frozenset(obj))

        return frozenset(result)

    @staticmethod
    def fgpartition(grid: Grid) -> Objects:
        """Partition grid by color excluding background"""
        bg = ARCDSL.mostcolor(grid)
        colors = ARCDSL.palette(grid) - {bg}
        result = set()
        h, w = len(grid), len(grid[0])

        for color in colors:
            obj = set()
            for i in range(h):
                for j in range(w):
                    if grid[i][j] == color:
                        obj.add((color, (i, j)))
            if obj:
                result.add(frozenset(obj))

        return frozenset(result)

    @staticmethod
    def toobject(patch: Patch, grid: Grid) -> Object:
        """Convert patch to object using grid colors"""
        h, w = len(grid), len(grid[0])
        result = set()
        for i, j in patch:
            if 0 <= i < h and 0 <= j < w:
                result.add((grid[i][j], (i, j)))
        return frozenset(result)

    @staticmethod
    def asobject(grid: Grid) -> Object:
        """Convert entire grid to object"""
        result = set()
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                result.add((val, (i, j)))
        return frozenset(result)

    @staticmethod
    def normalize(obj: Object) -> Object:
        """Move object to origin (0,0)"""
        if not obj:
            return obj

        # Find min coordinates
        min_i = min(loc[0] for _, loc in obj)
        min_j = min(loc[1] for _, loc in obj)

        # Shift to origin
        return frozenset((val, (i - min_i, j - min_j)) for val, (i, j) in obj)

    @staticmethod
    def shift(obj: Object, offset: Tuple[int, int]) -> Object:
        """Shift object by offset"""
        di, dj = offset
        return frozenset((val, (i + di, j + dj)) for val, (i, j) in obj)

    @staticmethod
    def shape(obj: Object) -> Tuple[int, int]:
        """Get bounding box dimensions of object"""
        if not obj:
            return (0, 0)

        locs = [loc for _, loc in obj]
        min_i = min(i for i, j in locs)
        max_i = max(i for i, j in locs)
        min_j = min(j for i, j in locs)
        max_j = max(j for i, j in locs)

        return (max_i - min_i + 1, max_j - min_j + 1)

    # =====================================
    # Group 4: Fill/Paint Operations (6)
    # =====================================

    @staticmethod
    def fill(grid: Grid, value: int, indices: Indices) -> Grid:
        """Fill cells at indices with value"""
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        for i, j in indices:
            if 0 <= i < h and 0 <= j < w:
                result[i][j] = value

        return tuple(tuple(row) for row in result)

    @staticmethod
    def paint(grid: Grid, obj: Object) -> Grid:
        """Paint object onto grid"""
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        for val, (i, j) in obj:
            if 0 <= i < h and 0 <= j < w:
                result[i][j] = val

        return tuple(tuple(row) for row in result)

    @staticmethod
    def underfill(grid: Grid, value: int, indices: Indices) -> Grid:
        """Fill only background cells"""
        bg = ARCDSL.mostcolor(grid)
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        for i, j in indices:
            if 0 <= i < h and 0 <= j < w:
                if result[i][j] == bg:
                    result[i][j] = value

        return tuple(tuple(row) for row in result)

    @staticmethod
    def underpaint(grid: Grid, obj: Object) -> Grid:
        """Paint object only on background"""
        bg = ARCDSL.mostcolor(grid)
        h, w = len(grid), len(grid[0])
        result = [list(row) for row in grid]

        for val, (i, j) in obj:
            if 0 <= i < h and 0 <= j < w:
                if result[i][j] == bg:
                    result[i][j] = val

        return tuple(tuple(row) for row in result)

    @staticmethod
    def cover(grid: Grid, obj: Object) -> Grid:
        """Remove object from grid (fill with background)"""
        bg = ARCDSL.mostcolor(grid)
        indices = frozenset(loc for _, loc in obj)
        return ARCDSL.fill(grid, bg, indices)

    @staticmethod
    def canvas(value: int, dims: Tuple[int, int]) -> Grid:
        """Create blank grid"""
        h, w = dims
        return tuple(tuple(value for _ in range(w)) for _ in range(h))

    # =====================================
    # Group 5: Geometric Operations (4)
    # =====================================

    @staticmethod
    def crop(grid: Grid, start: Tuple[int, int], dims: Tuple[int, int]) -> Grid:
        """Crop subgrid"""
        si, sj = start
        h, w = dims
        result = []

        for i in range(si, min(si + h, len(grid))):
            if i >= 0 and i < len(grid):
                row = []
                for j in range(sj, min(sj + w, len(grid[0]))):
                    if j >= 0 and j < len(grid[0]):
                        row.append(grid[i][j])
                if row:
                    result.append(tuple(row))

        return tuple(result) if result else ((0,),)

    @staticmethod
    def subgrid(obj: Object, grid: Grid) -> Grid:
        """Get minimal subgrid containing object"""
        if not obj:
            return ((0,),)

        locs = [loc for _, loc in obj]
        min_i = min(i for i, j in locs)
        max_i = max(i for i, j in locs)
        min_j = min(j for i, j in locs)
        max_j = max(j for i, j in locs)

        return ARCDSL.crop(grid, (min_i, min_j), (max_i - min_i + 1, max_j - min_j + 1))

    @staticmethod
    def center(obj: Object) -> Tuple[int, int]:
        """Get center of object"""
        if not obj:
            return (0, 0)

        locs = [loc for _, loc in obj]
        min_i = min(i for i, j in locs)
        max_i = max(i for i, j in locs)
        min_j = min(j for i, j in locs)
        max_j = max(j for i, j in locs)

        return ((min_i + max_i) // 2, (min_j + max_j) // 2)

    @staticmethod
    def corners(obj: Object) -> Indices:
        """Get corner positions of object bounding box"""
        if not obj:
            return frozenset()

        locs = [loc for _, loc in obj]
        min_i = min(i for i, j in locs)
        max_i = max(i for i, j in locs)
        min_j = min(j for i, j in locs)
        max_j = max(j for i, j in locs)

        return frozenset([(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)])

    # =====================================
    # Group 6: Index Operations (2)
    # =====================================

    @staticmethod
    def asindices(grid: Grid) -> Indices:
        """Get all grid indices"""
        h, w = len(grid), len(grid[0])
        return frozenset((i, j) for i in range(h) for j in range(w))

    @staticmethod
    def ofcolor(grid: Grid, value: int) -> Indices:
        """Get indices of specific color"""
        result = set()
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == value:
                    result.add((i, j))
        return frozenset(result)

    # =====================================
    # Group 7: Advanced Operations (2)
    # =====================================

    @staticmethod
    def occurrences(grid: Grid, pattern: Object) -> Indices:
        """Find all occurrences of pattern in grid"""
        if not pattern:
            return frozenset()

        # Normalize pattern to origin
        norm_pattern = ARCDSL.normalize(pattern)
        ph, pw = ARCDSL.shape(norm_pattern)
        h, w = len(grid), len(grid[0])

        result = set()

        # Try each position
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                # Check if pattern matches at this position
                shifted = ARCDSL.shift(norm_pattern, (i, j))
                matches = True
                for val, (pi, pj) in shifted:
                    if not (0 <= pi < h and 0 <= pj < w):
                        matches = False
                        break
                    if grid[pi][pj] != val:
                        matches = False
                        break

                if matches:
                    result.add((i, j))

        return frozenset(result)

    @staticmethod
    def compress(grid: Grid) -> Grid:
        """Remove uniform rows and columns"""
        h, w = len(grid), len(grid[0])

        # Find uniform rows
        uniform_rows = set()
        for i in range(h):
            if len(set(grid[i])) == 1:
                uniform_rows.add(i)

        # Find uniform columns
        uniform_cols = set()
        for j in range(w):
            col = [grid[i][j] for i in range(h)]
            if len(set(col)) == 1:
                uniform_cols.add(j)

        # Build compressed grid
        result = []
        for i in range(h):
            if i not in uniform_rows:
                row = []
                for j in range(w):
                    if j not in uniform_cols:
                        row.append(grid[i][j])
                if row:
                    result.append(tuple(row))

        return tuple(result) if result else ((0,),)


# =====================================
# Convenience functions
# =====================================

def grid_from_list(lst: List[List[int]]) -> Grid:
    """Convert list to Grid type"""
    return tuple(tuple(row) for row in lst)


def grid_to_list(grid: Grid) -> List[List[int]]:
    """Convert Grid to list"""
    return [list(row) for row in grid]


# =====================================
# Testing
# =====================================

if __name__ == "__main__":
    # Test basic operations
    test_grid = grid_from_list([
        [0, 1, 0],
        [2, 0, 3],
        [0, 4, 0]
    ])

    print("Original:")
    for row in test_grid:
        print(row)

    print("\nRotated 90:")
    rot = ARCDSL.rot90(test_grid)
    for row in rot:
        print(row)

    print("\nObjects:")
    objs = ARCDSL.objects(test_grid)
    print(f"Found {len(objs)} objects")

    print("\nPalette:", ARCDSL.palette(test_grid))
    print("Most color:", ARCDSL.mostcolor(test_grid))
    print("Least color:", ARCDSL.leastcolor(test_grid))

    print("\nDSL V1 Ready - 40 Primitives Implemented!")