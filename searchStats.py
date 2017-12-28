import pstats
p = pstats.Stats('searchStats')
p.strip_dirs().sort_stats('cumulative').print_stats()
