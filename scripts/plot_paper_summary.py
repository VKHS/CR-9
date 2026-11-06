#!/usr/bin/env python3
"""Render frozen-rule reference outcomes with their recorded or identity-derived costs.

This plots the manuscript registry; it never trains or substitutes benchmark
measurements. Requires the optional `plots` dependencies. Outputs PNG and SVG.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paper_common import DATASETS, load_registered, load_reported, peptides_confirmation_caption, resolved_policy_view


def render(output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    data, registered = load_reported(), load_registered()
    effects = {d: resolved_policy_view(data, d, registered) for d in DATASETS}
    costs = {d: effects[d]['costs'] for d in DATASETS}
    ledger = data['supplementary_table_S23_assurance_and_expected_compute']
    order = DATASETS
    ink, muted = '#14283f', '#526477'
    palette = {'radius':'#157f89', 'MPC-RH':'#7456a8', 'qC returned':'#748292'}
    matplotlib.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'axes.edgecolor':'#c4ced8', 'axes.labelcolor':ink, 'text.color':ink,
        'xtick.color':muted, 'ytick.color':ink, 'svg.fonttype':'none', 'svg.hashsalt':'cross-morse-paper-summary'})
    fig, axes = plt.subplots(1,2,figsize=(14.4,6.9),gridspec_kw={'width_ratios':[1.28,1]},sharey=True)
    fig.subplots_adjust(left=.15,right=.965,top=.74,bottom=.22,wspace=.20)
    fig.set_facecolor('white')
    fig.text(.035,.94,'Policy outcomes and inference work',fontsize=23,fontweight='bold')
    fig.text(.035,.895,'Seven datasets · frozen-rule reference outcomes · comparison with the selected open-loop rule',fontsize=12,color=muted)
    left,right=axes
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x',color='#e8edf1',linewidth=.8)
        for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=12)
        ax.set_ylim(-.65,6.65)
    for index,dataset in enumerate(order):
        y=len(order)-1-index
        effect,cost=effects[dataset],costs[dataset]
        color=palette.get(effect['deployed_rule'],palette['qC returned'])
        if effect['loss_effect_nats'] is None:
            left.text(0,y,'final outcome unavailable',ha='right',va='center',fontsize=9,color=muted)
            continue
        point=1000*effect['loss_effect_nats']
        interval=effect['loss_effect_interval_nats']
        if interval:
            lo,hi=[1000*x for x in interval]
            left.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color=color,
                          markersize=8,capsize=4,elinewidth=2,zorder=3)
            left.text(1.15,y,f"{point:.2f}",ha='right',va='center',fontsize=10,color=color)
        else:
            left.scatter([0],[y],marker='D',color=color,s=48,zorder=3)
            left.annotate('comparator retained',(0,y),xytext=(-8,10),textcoords='offset points',
                          ha='right',fontsize=9,color=muted)
        branch=ledger[dataset]
        p=branch['fallback_probability']
        expected=1.0 if effect['comparator_identity'] else ((1-p)*branch['controller_cost']+p*branch['comparator_cost'])/branch['comparator_cost']
        assurance=cost['C_assure']/cost['C_qC']
        right.plot([1,assurance],[y,y],color='#d6e0e7',linewidth=2,zorder=1)
        right.scatter([expected],[y],s=63,color=color,zorder=3)
        right.scatter([assurance],[y],s=65,facecolor='white',edgecolor=color,linewidth=1.8,zorder=2)
        if expected==assurance:
            right.text(expected+.055,y,'1.00×',va='center',fontsize=10,color=muted)
        else:
            fmt=lambda value: str(Decimal(str(value)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
            right.text(expected+.055,y+.02,fmt(expected)+'×',va='center',fontsize=10,color=color)
            right.text(assurance+.055,y,fmt(assurance)+'×',va='center',fontsize=10,color=muted)
    left.set_yticks(np.arange(len(order))[::-1],[f"{d}{'*' if effects[d].get('registered_confirmation_pass') is False else ''}  ({effects[d]['loss_name']})" for d in order])
    left.set_xlim(-14.5,1.5)
    left.set_xticks([-12,-9,-6,-3,0])
    left.axvline(0,color='#aab7c4',linewidth=1)
    left.set_xlabel('Final-policy loss effect versus qC (millinats)\nComparator-return points are exact identities',labelpad=12)
    left.set_title('A  Final-policy effect; reported intervals where available',loc='left',fontsize=12,fontweight='bold',pad=18)
    right.tick_params(axis='y',labelleft=False)
    right.set_xlim(.93,2.57)
    right.set_xticks([1,1.25,1.5,1.75,2,2.25,2.5],['1.00','1.25','1.50','1.75','2.00','2.25','2.50'])
    right.axvline(1,color='#aab7c4',linewidth=1)
    right.set_xlabel('Serial cost relative to qC (qC = 1.00)\nComparator returns execute the comparator once',labelpad=12)
    right.set_title('B  Expected and two-branch assurance cost',loc='left',fontsize=12,fontweight='bold',pad=18)
    policy_legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=color,markeredgecolor=color,
                   markersize=7,label=label) for label,color in [('Radius',palette['radius']),('Receding horizon',palette['MPC-RH']),('Comparator',palette['qC returned'])]]
    left.legend(handles=policy_legend,loc='lower left',bbox_to_anchor=(-.02,1.10),ncol=3,frameon=False,fontsize=10,
                handletextpad=.4,columnspacing=1.2)
    right.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=ink,markeredgecolor=ink,label='Expected'),
                          Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor=ink,label='Assurance')],
                 loc='lower left',bbox_to_anchor=(-.02,1.10),ncol=2,frameon=False,fontsize=10)
    fig.text(.035,.075,'Sources: main Tables 1–2 and SI S23. Comparator effects and costs follow identity; other values are reported references.',fontsize=10,color=muted)
    fig.text(.035,.041,'*' + peptides_confirmation_caption(data, registered),fontsize=10,color=muted)
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('svg','png'):
        meta={'Creator':'CROSS-MORSE scripts/plot_paper_summary.py'}
        if suffix=='svg': meta['Date']=None
        path = output/f'reported_results.{suffix}'
        fig.savefig(path,dpi=180,facecolor='white',metadata=meta)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    render(args.output_dir)
    print(args.output_dir/'reported_results.svg')
    print(args.output_dir/'reported_results.png')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
