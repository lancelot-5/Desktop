def bfs(game_map_in, pac_man, food):
    game_map = game_map_in.copy()
    lists = []
    lists.append(pac_man)
    for now in lists:
        # print(lists)
        # print(now)
        # 当前走到的节点是哪一个节点，也就是最后走的一步，是哪一步，去列表的最后的一个值就是索引-1
        if food in lists:
            print("出来了")
            break
        row, col = now
        # python 里的解构也叫解包 now包括两个位置，一个行，一个列
        game_map[row][col] = 2
        # 这个代表就是走过的点，为2，因为你走过的路是不能再走的，除了走不通返回的时候，也就是为了走不通按原来走过的路原路返回
        if (game_map[row - 1][col] == 0) & ([row - 1, col] not in lists):
            # 上方可以走
            lists.append([row - 1, col])
        if (game_map[row][col + 1] == 0) & ([row, col + 1] not in lists):
            # 右方可以走
            lists.append([row, col + 1])
        if (game_map[row + 1][col] == 0) & ([row + 1, col] not in lists):
            # 下方可以走
            lists.append([row + 1, col])
        if (game_map[row][col - 1] == 0) & ([row, col - 1] not in lists):
            # 左方可以走0
            lists.append([row, col - 1])
    return lists
