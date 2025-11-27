CHROM_SETS = {
    'test': ['chr14'],
    'eval': ['chr15'],
}
CHROM_SETS['train'] = [
    f'chr{i}' 
    for i in range(1, 23) 
    if f'chr{i}' not in (CHROM_SETS['test']+CHROM_SETS['eval'])
]
