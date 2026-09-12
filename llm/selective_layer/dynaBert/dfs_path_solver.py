#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFSで各頂点への経路を求めるPythonコード

行方法:
    python dfs_path_solver.py
"""


def dfs_path(graph, start):
    """
    DFSで始点から各頂点への経路を求める

    Parameters:
        graph: dict - 隣接リスト表現のグラフ
        start: int - 始点

    Returns:
        came_from: dict - 各頂点が「どの頂点から」発見されたか
        get_path: function - 頂点への経路を返す関数
        visit_order: list - DFSの訪問順
    """
    visited = set()
    stack = [start]
    came_from = {}
    visit_order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        visit_order.append(node)

        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
                if neighbor not in came_from:
                    came_from[neighbor] = node

    def get_path(node):
        """始点からnodeまでの経路を返す"""
        if node != start and node not in came_from:
            return None

        path = [node]
        current = node
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    return came_from, get_path, visit_order


# ============================================
# メイン処理
# ============================================
if __name__ == "__main__":

    # グラフの定義
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [1, 5],
        4: [2],
        5: [3]
    }

    print("=" * 50)
    print("DFSで各頂点への経路を求める")
    print("=" * 50)
    print()
    print("グラフ構造:")
    print("    1")
    print("   / \")
    print("  2   3")
    print(" /     \")
    print("4       5")
    print()
    print("隣接リスト:")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")
    print()

    # DFS実行
    came_from, get_path, visit_order = dfs_path(graph, start=1)

    print("=" * 50)
    print("結果")
    print("=" * 50)
    print()
    print(f"DFS訪問順: {' -> '.join(map(str, visit_order))}")
    print()

    print("発見元の記録:")
    for child, parent in came_from.items():
        print(f"  頂点{child} は 頂点{parent} から発見された")
    print()

    print("始点1から各頂点への経路:")
    for node in sorted(graph.keys()):
        path = get_path(node)
        if path:
            print(f"  1 → {node}: {' -> '.join(map(str, path))}")
        else:
            print(f"  1 → {node}: 到達不能")
