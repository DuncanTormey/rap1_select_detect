from __future__ import print_function
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_venn as v
import itertools
from itertools import combinations
from collections import defaultdict
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
sys.path = ['/home/dut/bin/python_scripts'] + sys.path
import lims_classes as lims


def get_colors(num_colors):
    colors = []
    k = 1
    dark = 25
    for i in np.arange(0., 360., 360. / num_colors):
        if k % 2 == 0:
            mod = 20
        else:
            dark -= 1
            mod = dark
        hue = i / 360.
        lightness = (mod + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        k += 1
    return colors

def ret_merge_single_bwa_align_cmd(sample, fastqs, outdir, cpus, fasta_ref_path,trimmomatic, trim_param):
    cpus = str(cpus)
    lane1 = fastqs[0].path
    lane2 = fastqs[1].path

    fastq = '%s/%s.fastq.gz' % (outdir, sample)

    trimmed_fastq = fastq.replace('.fastq.gz', '.trimmed.fastq.gz')

    merge = 'cat %s %s > %s' % (lane1, lane2, fastq)

    trim = 'java -jar %s SE -threads 8 %s %s %s &> %s' % (
        trimmomatic, fastq, trimmed_fastq, trim_param, trimmed_fastq.replace('.fastq.gz', '.out'))
    output_bam = '%s/%s.bam' % (outdir, sample)

    bwa_command = "bwa mem -M -t %s %s %s" % (cpus,
                                              fasta_ref_path,
                                              trimmed_fastq,
                                              )

    sam_to_sorted_bam = ('samtools view -Sbq 2 -@ %s - | '
                         'samtools sort -o -@ %s - %s > %s; samtools index %s') % (
                             cpus, cpus, sample + '_temp', output_bam, output_bam)

    align_sort_cmd = "(%s; %s; %s | %s)\n" % (
        merge, trim, bwa_command, sam_to_sorted_bam)

    return align_sort_cmd, output_bam


def ret_pysamstats_cmd(sample, alignment, outdir, fasta_ref_path):
    output_table = '%s/%s_position_data.tsv' % (outdir, sample)
    fields = 'chrom,pos,ref,reads_all,matches,mismatches,deletions,insertions,A,C,T,G,N'
    cmd = 'pysamstats -f %s --type variation --pad -D 100000000 --fields=%s %s | tr -d \'\r\' > %s' % (fasta_ref_path,
                                                                                                       fields,
                                                                                                       alignment,
                                                                                                       output_table)
    return cmd, output_table


def align_from_single_lims_report(data, outdir, reference, trimmomatic, trim_param):
    alignments = {}
    for sample in data.indexed_sample_reports:
        command, alignment = ret_merge_single_bwa_align_cmd(sample,
                                                            data.indexed_sample_reports[
                                                                sample],
                                                            outdir,
                                                            8,
                                                            reference,
                                                            trimmomatic,
                                                            trim_param)
        alignments[sample] = alignment
        if not os.path.exists(alignment):
            subprocess.call(command, shell=True)

    return alignments


def ret_position_tables_from_alignments(alignments, outdir, fasta_ref_path):
    position_data_tables = {}
    for sample in alignments:
        command, position_data_table = ret_pysamstats_cmd(
            sample, alignments[sample], outdir, fasta_ref_path)
        position_data_tables[sample] = position_data_table
        if not os.path.exists(position_data_table):
            subprocess.call(command, shell=True)

    return position_data_tables


def norm_position_df(position_df):
    columns_to_norm = ['matches', 'mismatches',
                       'deletions', 'insertions', 'A', 'T', 'C', 'G', 'N']
    non_normed = [x for x in position_df.columns if x not in columns_to_norm]
    normed_columns = position_df[columns_to_norm].divide(
        position_df['reads_all'], axis='rows')
    normed_columns = normed_columns * 100
    normed_df = pd.concat([position_df[non_normed],
                           normed_columns], axis=1,
                          join_axes=[position_df.index])
    return normed_df


def merge_data_frames(position_data_tables, time_course_samples, dropped=['chrom', 'pos', 'ref'], merged_df=pd.DataFrame()):
    for sample in time_course_samples:
        sample_df = pd.read_csv(position_data_tables[sample], sep='\t')
        sample_df = norm_position_df(sample_df)
        if merged_df.empty:
            sample_df.columns = ['%s-%s' % (
                sample, x) if x not in dropped else x for x in sample_df.columns]
            merged_df = sample_df
        else:
            sample_df.drop(dropped, axis=1, inplace=True)
            sample_df.columns = ['%s-%s' %
                                 (sample, x) for x in sample_df.columns]
            merged_df = pd.concat([merged_df, sample_df],
                                  axis=1, join_axes=[merged_df.index])
    return merged_df


def compute_deltas(merged_df, global_columns=['chrom', 'pos', 'ref']):
    sample_suffixes = list(set([x.split('_')[0] for x in merged_df.columns]))
    fields = list(set([x.split('-')[-1]
                       for x in merged_df.columns if x not in global_columns]))
    for sample in sample_suffixes:
        sample_columns = [x for x in merged_df.columns if x.split('_')[
            0] == sample]
        for field in fields:
            sample_fields = [
                x for x in sample_columns if x.split('-')[-1] == field]
            sample_fields = sorted(sample_fields,
                                   key=lambda x: int(x.split('_')[1][1]),
                                   reverse=True)
            combos = combinations(sample_fields, 2)
            for combo in combos:
                rnd1 = combo[0].split('-')[0].split('_')[-1]
                rnd2 = combo[1].split('-')[0].split('_')[-1]

                key = '%s_%s%s-d_%s' % (sample, rnd2, rnd1, field)
                merged_df[key] = merged_df[combo[1]] - merged_df[combo[0]]
                key = '%s_%s%s-fc_%s' % (sample, rnd2, rnd1, field)
                merged_df[key] = merged_df[combo[1]] / merged_df[combo[0]]

    return merged_df


def make_round_zero_comparisons(df, samples=['A', 'B', 'C']):
    fc_cols = []
    for s in samples:
        r1 = 'ABC_R0-mismatches'
        r2 = '%s_R5-mismatches' % s
        fc = '%s_R0R5-fc_mismatches' % s
        fc_cols.append(fc)
        df[fc] = (df[r1] / df[r2]) * -1
    mean = 'avg_R0R5-fc_mismatches'
    std = 'std_R0R5-fc_mismatches'
    df[mean] = df[fc_cols].replace(
        -np.inf, np.nan).mean(axis=1, skipna=True)
    df[std] = df[fc_cols].replace(
        -np.inf, np.nan).std(axis=1, skipna=True)

    return df, mean, std


def make_subset_df(merged_df, kept_columns, avg_column_regexs, positions):
    stat_df = pd.DataFrame()
    for column in kept_columns:
        if column in merged_df.columns:
            stat_df[column] = merged_df[column]
            stat_df[column] = merged_df[column]
        else:
            print('Column NOT found', column)
    for column_regex in avg_column_regexs:
        df = merged_df.filter(regex=(column_regex))
        stat_df[
            'mean_' + column_regex] = df.mean(axis=1)
        stat_df[
            'stdev_' + column_regex] = df.std(axis=1)
    stat_df = stat_df[stat_df['pos'].isin(positions)]
    return stat_df


def row_column_checker(a):
    for i in range(len(a.columns) - 1):
        if a.iloc[0, i] < a.iloc[0, i + 1]:
            return False
    return True


def get_shared_decreasing_positions(df):
    position_dict = defaultdict(list)
    for a in ['A', 'B', 'C']:
        for position in df['pos']:
            x = df[df['pos'] == position]
            x_values = x.filter(regex=('%s_R.*-mismatches' % a))
            x_values = x_values.sort_index(axis=1)
            if row_column_checker(x_values):
                position_dict[a].append(position)
        # print(len(position_dict[a]))
    union = set(position_dict.values()[0]).intersection(
        *position_dict.values()[1:])
    # print(len(union))
    return union


def get_shared_r_zero_decreased_positions(df):
    position_dict = {}
    for a in ['A', 'B', 'C']:
        r1 = 'ABC_R0-mismatches'
        r2 = '%s_R5-mismatches' % a
        temp = df[df[r1] > df[r2]]
        pos = list(temp['pos'])
        position_dict[a] = pos
    union = set(position_dict.values()[0]).intersection(
        *position_dict.values()[1:])
    # print(len(union))
    return union


def add_protein_data(df):
    nuc = ['A', 'T', 'G', 'C']
    a_cod = []
    t_cod = []
    g_cod = []
    c_cod = []
    a_aa = []
    t_aa = []
    g_aa = []
    c_aa = []
    codon_positions = []
    i = 0
    for n in df['ref']:
        if i % 3 == 0:
            codon = Seq(''.join(list(df['ref'][i:i + 3])), generic_dna)

        k = i % 3
        alt_cod = [codon[:k] + j + codon[k + 1:] for j in nuc]
        alt_aa = [str(c.translate()) for c in alt_cod]
        alt_cod = [str(_) for _ in alt_cod]
        codon_position = k + 1
        a_cod.append(alt_cod[0])
        t_cod.append(alt_cod[1])
        g_cod.append(alt_cod[2])
        c_cod.append(alt_cod[3])
        a_aa.append(alt_aa[0])
        t_aa.append(alt_aa[1])
        g_aa.append(alt_aa[2])
        c_aa.append(alt_aa[3])
        codon_positions.append(codon_position)
        i += 1

    df['a_cod'] = a_cod
    df['t_cod'] = t_cod
    df['g_cod'] = g_cod
    df['c_cod'] = c_cod
    df['a_aa'] = a_aa
    df['t_aa'] = t_aa
    df['g_aa'] = g_aa
    df['c_aa'] = c_aa
    df['codon_positions'] = codon_positions

    return df

##############################################################################
####################### plotting functions ###################################
##############################################################################


def plot_single_rep_mutation_rate(a, title):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(1, figsize=(12, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(a['pos'], a['mismatches'], color='green')

    x_ticks = np.arange(0, a['pos'].max() + 25, 25)
    y_ticks = np.arange(0, a['mismatches'].max() + 2, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_ylabel('Percent Mismatches')
    ax.set_xlabel('Position')
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig('../fig/%s_perc_mismatches.png' % title)
    plt.show()
    return fig, ax

def plot_single_rep_coverage(a, title):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(1, figsize=(12, 3), dpi = 300)
    ax = fig.add_subplot(111)
    ax.plot(a['pos'], a['reads_all'], color='orange')

    x_ticks = np.arange(0, a['pos'].max() + 25, 25)
    y_ticks = np.arange(0, a['reads_all'].max() + 10000, 10000)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_ylabel('Coverage')
    ax.set_xlabel('Position')
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig('../fig/%s_perc_coverage.png' % title)
    plt.show()
    return fig, ax


def plot_time_course_mismatches(df, exp = 'A'):
    plt.close()
    union = get_shared_decreasing_positions(df)
    colors = get_colors(len(union))
    marker = itertools.cycle(
        ('--o', '--v', '--^', '--<', '-->', '--8', '--s', '--p', '--h', '--H', '--D', '--d'))
   
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(1, figsize=(17, 12), dpi = 300)
    ax = fig.add_subplot(111)
    i = 0
    for position in df['pos']:
        x = df[df['pos'] == position]
        x_values = x.filter(regex=('%s_R.*-mismatches' % exp))
        x_values = x_values.sort_index(axis=1)
        if row_column_checker(x_values) and position in union:
            ax.plot(np.arange(len(x_values.columns)),
                    x_values.values[0], marker.next(),
                    color=colors[i], label=position,
                    lw=1, markersize=7, markeredgewidth=0.5, alpha=0.5)
    
            i += 1
    ax.set_xticks(np.arange(len(x_values.columns)))
    ax.set_xticklabels([x.split('-')[0].split('_')[1]
                        for x in x_values.columns])
    ax.set_xlim(-0.1, len(x_values.columns) - 1 + 0.1)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel('Percent Mismatch(%)', fontsize=24)
    ax.set_xlabel('Round', fontsize=24)
    ax.set_title(
        'Mismatch Percentage vs Round For Shared Decreasing Positions: %s' % exp, fontsize=30)
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)
    
    fig.savefig('../fig/%s_shared_mismatch_positions.png' %
                exp, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.close()
    return fig,ax

def plot_shared_decreasing_ven(df):
    plt.close()
    position_dict = defaultdict(list)
    for a in ['A', 'B', 'C']:
        for position in df['pos']:
            x = df[df['pos'] == position]
            x_values = x.filter(regex=('%s_R.*-mismatches' % a))
            x_values = x_values.sort_index(axis=1)
            if row_column_checker(x_values):
                position_dict[a].append(position)
    for k in position_dict:
        print(len(position_dict[k]))
    union = set(position_dict.values()[0]).intersection(
        *position_dict.values()[1:])
    print(len(union))
    fig = plt.figure(1, figsize=(10, 10), dpi = 300)
    plt.style.use('seaborn-whitegrid')
    matplotlib.rcParams.update({'font.size': 20})
    v.venn3([set(l) for l in position_dict.values()],
            position_dict.keys(), alpha=0.4)
    plt.title('Shared Decreasing Positions', fontsize=24)
    fig.savefig('../fig/shared_decreasing_position_ven.png')
    # plt.show()
    #plt.close()
    return fig

def plot_shared_decreased_ven(df):
    plt.close()
    position_dict = {}
    for a in ['A', 'B', 'C']:
        r1 = 'ABC_R0-mismatches'
        r2 = '%s_R5-mismatches' % a
        temp = df[df[r1] > df[r2]]
        pos = list(temp['pos'])
        position_dict[a] = pos
    fig = plt.figure(1, figsize=(10, 10), dpi = 300)
    plt.style.use('seaborn-whitegrid')
    matplotlib.rcParams.update({'font.size': 20})
    v.venn3([set(l) for l in position_dict.values()],
            position_dict.keys(), alpha=0.4)
    plt.title('Shared Decreased Positions', fontsize=24)
    fig.savefig('../fig/shared_decreased_position_ven.png')
    # plt.show()
    return fig
    
def plot_average_mutation(position_data_tables, sample_list, rounds=5, exp = 'A'):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    
    samples = [s for s in sample_list if s[0:2] == '%s_' % exp]
    avgs = []
    boxes = []
    stdevs = []
    ps = []
    fig = plt.figure(1, figsize=(10, 10), dpi = 300)
    ax = fig.add_subplot(111)
    for i in range(1, rounds + 1, 1):
        sample = [s for s in samples if int(s[3]) == i]
        df = pd.read_csv(position_data_tables[sample[0]], sep='\t')
        df = df[df['pos'] > 73]
        df = df[df['pos'] < 841]
        df = norm_position_df(df)
        avgs.append(df['mismatches'].mean())
        stdevs.append(df['mismatches'].std())
        ps.append(i)
        boxes.append(df['mismatches'])
    bp = ax.boxplot(boxes, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(color='black', linewidth=1.2)
        patch.set(facecolor='blue', alpha=0.5)

    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)

    for median in bp['medians']:
        median.set(color='black', linewidth=1.2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='lightgreen', alpha=0.5, markersize=5)
    ax.set_ylabel('Percent Mismatch(%)', fontsize=20)
    ax.set_xlabel('Round', fontsize=20)
    ax.set_title(
        'Distribution of Percent Mismatch Experiment: %s' % exp, fontsize=24)
    fig.tight_layout()
    fig.savefig('../fig/time_course_distributions_%s.png' % exp)
    # mpld3.show()
    # plt.show()
    #plt.close()
    return fig, ax

def plot_avg_fc(df):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(1, figsize=(12, 7), dpi = 300)
    ax = fig.add_subplot(111)
    width = 0.45
    pos = np.arange(len(df))
    avg = df['avg_R0R5-fc_mismatches'].tolist()
    std = df['std_R0R5-fc_mismatches'].tolist()
    xlabels = df['pos'].tolist()
    ax.bar(pos, avg, width, color='blue', yerr=std,
           alpha=0.5, error_kw={'ecolor': 'black', 'linewidth': 1, 'capsize': 1})
    ax.set_xticks(pos)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis='both', which='major', labelsize=18)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xlim(-0.5, len(df) + 0.25)
    ax.set_title('Top 25 Average Fold Change R0 vs R5', fontsize=24)
    ax.set_xlabel('Position', fontsize=20)
    ax.set_ylabel('Fold Change', fontsize=20)
    fig.tight_layout()
    fig.savefig('../fig/top_25_average_fc.png')
    #plt.close()
    return fig, ax

def plot_position_base_drop(position, df, exp = 'A'):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(1, figsize=(17, 12), dpi = 300)
    ax = fig.add_subplot(111)
    for b in ['A', 'C', 'G', 'T']:
        x = df[df['pos'] == position]
        if b != x['ref'].values[0]:
            x_values = x.filter(regex=('.*%s.*_R.*-%s' % (exp, b)))
            print(x_values)
            x_values = x_values.sort_index(axis=1)
            ax.plot(np.arange(len(x_values.columns)),
                    x_values.values[0], label=b,
                    lw=1, alpha=0.5)

    ax.set_xticks(np.arange(len(x_values.columns)))
    ax.set_xticklabels([x.split('-')[0].split('_')[1]
                        for x in x_values.columns])
    ax.set_xlim(-0.1, len(x_values.columns) - 1 + 0.1)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylabel('%', fontsize=24)
    ax.set_xlabel('Round', fontsize=24)
    ax.set_title(
        'Experiment %s Position: %s' % (exp, position), fontsize=30)
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc=2)

    fig.savefig('../fig/%s_%s_base_drop.png' %
                (position, a), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.close()
    return fig, ax

##########################################################################
# ################################## ANAYSIS #############################
# ########################################################################


if __name__ == '__main__':
    molng = ['/n/analysis/Baumann/lpn/MOLNG-1608/']
    samples = ["ABC_R0", "A_R1", "A_R2", "A_R3", "A_R4", "A_R5", "B_R1", "B_R2",
               "B_R3", "B_R4", "B_R5", "C_R1", "C_R2", "C_R3", "C_R4", "C_R5",
               "XYZ_R0", "Mut_lib_plasmid"]
    
    time_course_samples = ["ABC_R0", "A_R1", "A_R2", "A_R3", "A_R4", "A_R5",
                           "B_R1", "B_R2", "B_R3", "B_R4", "B_R5", "C_R1",
                           "C_R2", "C_R3", "C_R4", "C_R5", "XYZ_R0"]
    starting_samples = ["ABC_R0", "XYZ_R0", "Mut_lib_plasmid"]
    
    trimmomatic = '/home/dut/bin/trinityrnaseq-2.0.6/trinity-plugins/Trimmomatic/trimmomatic-0.32.jar'
    trim_param = 'ILLUMINACLIP:/home/dut/bin/trinityrnaseq-2.0.6/trinity-plugins/Trimmomatic/adapters/NexteraPE-PE.fa:2:30:10 LEADING:30 TRAILING:30 SLIDINGWINDOW:4:30 MINLEN:36'
    # samples = time_course_samples
    flowcell = "HKK57BCXXa"
    reference = '/home/dut/projects/pombe/rap1/fasta/lili_PCR_for_NGS.fasta'
    alignment_outdir = '/home/dut/projects/pombe/rap1/alignments'
    tables_outdir = '/home/dut/projects/pombe/rap1/tables'
    data = lims.Indexed_sample_reports(molng, samples)
    data.select_flowcells([flowcell])
    alignments = align_from_single_lims_repert(
    data, alignment_outdir, reference)

    position_data_tables = ret_position_tables_from_alignments(
        alignments, tables_outdir, reference)
    
    plot_average_mutation(position_data_tables, time_course_samples)
    
    for sample in starting_samples:
        df = pd.read_csv(position_data_tables[sample], sep='\t')
        df = norm_position_df(df)
        summary = df.describe()
        summary.to_csv('../tables/%s_summary_stats_pos.tsv' %
                       sample, sep='\t')
        plot_single_rep_coverage(df, sample)
        plot_single_rep_mutation_rate(df, sample)
    
    x = merge_data_frames(position_data_tables)
    x = x[x['pos'] > 73]
    x = x[x['pos'] <= 841]
    
    x = add_protein_data(x)
    x.to_excel('../tables/annotated_rap1_mutation_data.xls')
    x.to_html('../tables/annotated_rap1_mutation_data.html')
    
    shared_positions = get_shared_decreasing_positions(x)
    
    y = x[x['pos'].isin(shared_positions)]
    y.to_excel('../tables/annotated_shared_decreasing_positions.xls')
    y.to_html('../tables/annotated_shared_decreasing_positions.html')
    
    zero_positions = []
    for c in 'ABC':
        r1 = '%s_R1-mismatches' % c
        r2 = '%s_R5-mismatches' % c
        z = x[x[r2] == 0.0]
        zero_positions.extend(list(z['pos']))
    
    zero_positions = set(zero_positions)
    z = x[x['pos'].isin(zero_positions)]
    z.to_excel('../tables/annotated_zero_ending_positions.xls')
    z.to_html('../tables/annotated_zero_ending_positions.html')
    
    r_zero_decreased_positions = get_shared_r_zero_decreased_positions(x)
    
    plot_shared_decreasing_ven(x)
    plot_shared_decreased_ven(x)
    plot_time_course_mismatches(x)
    
    x, mean_fc, std_fc = make_round_zero_comparisons(x)
    
    x = x.sort(mean_fc, ascending=True)
    top_25 = x.head(n=25)
    top_25.to_excel('../tables/top_25_fc.xls')
    top_25.to_html('../tables/top_25_fc.html')
    plot_avg_fc(top_25)
    # for position in
    plot_position_base_drop(750, x)
