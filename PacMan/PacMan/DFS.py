def dfs(game_map_in, pac_man, food):
    game_map = game_map_in.copy()
    lists = []
    lists.append(pac_man)
    print(food)
    print(game_map)
    while lists:
        # print(game_map_in)
        # print(list)
        # 当前走到的节点是哪一个节点，也就是最后走的一步，是哪一步，去列表的最后的一个值就是索引-1
        now = lists[-1]
        if now == food:  # 如果现在的now等于我们之前定义的终点end
            print("BINGO")
            break
        row, col = now
        # python 里的解构也叫解包 now包括两个位置，一个行，一个列
        game_map[row][col] = 2
        # 这个代表就是走过的点，为2，因为你走过的路是不能再走的，除了走不通返回的时候，也就是为了走不通按原来走过的路原路返回
        if game_map[row - 1][col] == 0:
            # 上方可以走
            lists.append([row - 1, col])
            continue
        elif game_map[row][col + 1] == 0:
            # 右方可以走
            lists.append([row, col + 1])
            continue
        elif game_map[row + 1][col] == 0:
            # 下方可以走
            lists.append([row + 1, col])
            continue
        elif game_map[row][col - 1] == 0:
            # 左方可以走
            lists.append([row, col - 1])
            continue
        else:  # 走不通过，直接循环干掉每一步，重新调整路线
            lists.pop()
    # print(lists)
    # print(game_map)
    return lists