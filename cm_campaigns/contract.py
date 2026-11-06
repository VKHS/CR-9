"""Campaign coverage for the out-of-lifecycle experiments (E3-E6, E12).

Record-set names match records/specs.json.

Fourteen external rows means fourteen dataset–method comparisons, not fourteen
unique architecture names. Statistical units and accounting scopes stay explicit.
"""
DATASETS = ('ogbn-arxiv','ogbn-proteins','ogbn-products','Roman-empire',
            'PascalVOC-SP','COCO-SP','Peptides-func')
SLUGS = dict(zip(DATASETS, ('arxiv','proteins','products','roman_empire','pascalvoc_sp','coco_sp','peptides_func')))
ADMITTED = tuple(x for x in DATASETS if x != 'ogbn-products')
EXTERNAL = {
    'ogbn-arxiv': ('AMP','ADMP-GNN'),
    'ogbn-proteins': ('AMP','GRAMA'),
    'ogbn-products': ('ADMP-GNN','GRAMA'),
    'Roman-empire': ('Early-Exit GNN','ADMP-GNN'),
    'PascalVOC-SP': ('Graph Mamba','GRAMA'),
    'COCO-SP': ('Graph Mamba','GRAMA'),
    'Peptides-func': ('Graph Mamba','GRAMA'),
}
METRICS = {'ogbn-arxiv':'accuracy','ogbn-proteins':'rocauc','ogbn-products':'accuracy',
           'Roman-empire':'accuracy','PascalVOC-SP':'macro_f1','COCO-SP':'macro_f1','Peptides-func':'ap'}
UNITS = {'ogbn-arxiv':'cluster','ogbn-proteins':'cluster','ogbn-products':'cluster',
         'Roman-empire':'official_split','PascalVOC-SP':'graph','COCO-SP':'graph','Peptides-func':'graph'}
RECORD_SETS = {
    'baselines': {'tables':['S15','S15b'], 'datasets':list(DATASETS), 'analyzer':'baselines'},
    'budget_matched': {'tables':['S46','S46b'], 'datasets':list(ADMITTED), 'analyzer':'budget_matched'},
    'input_gate': {'tables':['S23'], 'datasets':list(ADMITTED), 'analyzer':'input_gate'},
    'factorials': {'tables':['S16','S16b'], 'datasets':['ogbn-arxiv','PascalVOC-SP'], 'analyzer':'factorials'},
    'negative_controls': {'tables':['S19'], 'datasets':[], 'dataset_scope':'declared',
        'coverage_keys':['control'], 'coverage_rows':[
            ['graph irrelevant'], ['misleading graph'], ['sensor bias'], ['support shift']],
        'analyzer':'negative_controls'},
    'nominal_time': {'tables':['S23b'], 'datasets':list(ADMITTED), 'analyzer':'nominal_time'},
    'sensor_parameters': {'tables':['S6c'], 'datasets':list(ADMITTED), 'analyzer':'sensor_parameters'},
    'cross_backbone': {'tables':['S25'], 'datasets':['ogbn-arxiv','Roman-empire','PascalVOC-SP'],
        'coverage_keys':['regime','model'], 'coverage_rows':[
            ['arxiv / homophily','residual MLP4-512'], ['arxiv / homophily','frozen GCNII'],
            ['Roman / heterophily','heterophily encoder4-384'], ['Roman / heterophily','frozen GCNII'],
            ['Pascal / independent long range','GraphGPS-style 6x128'],
            ['Pascal / independent long range','frozen GINE/GRAMA-style']],
        'analyzer':'cross_backbone'},
    'ensemble_grid': {'tables':['S26b'], 'datasets':list(ADMITTED), 'analyzer':'ensemble_grid'},
    'inserted_site': {'tables':['S45'], 'datasets':['PascalVOC-SP','COCO-SP','Peptides-func'], 'analyzer':'inserted_site'},
}
RUN_TABLES = ['1','2','S4','S12','S12c','S13','S14','S23','S41']
RUN_ROLES = ('configuration','environment','split_manifest','seed_manifest','checkpoint',
             'prediction','trajectory','statistical_output','code_snapshot','accounting',
             'job_log','parameter_inventory','sensor_trace','selector_trace','fixed_schedule')
SET_ROLES = ('configuration','environment','split_manifest','seed_manifest','raw_observations',
             'statistical_output','code_snapshot','job_log')
BASELINE_SEEDS = 12
HPO_TRIALS = 48
ARMS = ('E3','E4','E5','E6')

# Required fields for each supplemental reducer. Every file is hash-sealed;
# record-specific study contracts declare conditions, pairing and data roles.
COLUMNS = {
    'input_gate': ('dataset','seed','unit_id','g_input','comparator_work','controller_work','gate_work','actual_work','fallback_predictions_equal'),
    'factorials': ('dataset','seed','unit_id','cell','depth','loss'),
    'negative_controls': ('dataset','seed','unit_id','condition','comparator_loss','deployed_loss','fallback'),
    'nominal_time': ('dataset','seed','unit_id','repeat','warmup','seconds'),
    'sensor_parameters': ('dataset','seed','unit_id','depth','hidden_dim','sketch_width','sensing_period','sensed','action_id','sparse_products'),
    'cross_backbone': ('dataset','seed','unit_id','backbone','comparator_loss','deployed_loss'),
    'ensemble_grid': ('dataset','seed','unit_id','ensemble','loss'),
    'inserted_site': ('dataset','seed','unit_id','arm','comparator_loss','loss'),
}

def coverage_contract():
    return {'schema':'cross-morse-publication-contract-1','datasets':list(DATASETS),
            'official_run_tables':RUN_TABLES, 'record_sets':RECORD_SETS,
            'external_comparisons':{k:list(v) for k,v in EXTERNAL.items()},
            'baseline_seed_count':BASELINE_SEEDS,'hpo_trial_count':HPO_TRIALS,
            'budget_arms':list(ARMS),'run_required_roles':list(RUN_ROLES),
            'status_semantics':'A source/test pass is not a deposited-benchmark pass. Missing evidence is a blocking requirement.'}
