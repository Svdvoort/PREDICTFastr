try:
    import matplotlib.pyplot as plt
    from matplotlib2tikz import save as tikz_save
except ImportError:
    print("[PREDICT Warning] Cannot use plot_ROC function, as _tkinter is not installed")
import numpy as np
import json
import sys


def barplot(performance_json, figwidth=20):
    with open(performance_json, 'r') as fp:
        params = json.load(fp)

    params = params['Parameters']
    # output = paracheck(params)

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    groups = ['Histogram', 'Shape', 'Orientation', 'Semantic']
    ntimes_groups = [params['histogram_features'],
                     params['shape_features'],
                     params['orientation_features'],
                     params['semantic_features']]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_figwidth(figwidth)
    fig.set_figheight(figwidth/2)
    ax.set_xlim(0, 1)

    # Example data
    # groups = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    texture = ('All', 'LBP', 'GLCM', 'GLRLM', 'GLSZM', 'Gabor')
    ntimes_texture = [params['texture_all_features'],
                      params['texture_LBP_features'],
                      params['texture_GLCM_features'],
                      params['texture_GLRLM_features'],
                      params['texture_GLSZM_features'],
                      params['texture_Gabor_features']]
    y_pos = np.arange(len(groups))
    y_postick = np.arange(len(groups) + 1)
    # ntimes_groups = 3 + 10 * np.random.rand(len(groups))
    # error = np.random.rand(len(groups))

    fontsize = 20
    # Normal features
    colors = ['steelblue', 'lightskyblue']
    ax.barh(y_pos, ntimes_groups, align='center',
            color=colors[0], ecolor='black')
    ax.set_yticks(y_postick)
    ax.set_yticklabels(groups + ['Texture'])
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage', fontsize=fontsize)

    # Texture features
    left = 0
    y_pos = np.max(y_pos) + 1

    j = 0
    for i in np.arange(len(texture)):
        color = colors[j]
        if j == 0:
            j = 1
        else:
            j = 0
        ax.barh(y_pos, ntimes_texture[i], align='center',
                color=color, ecolor='black', left=left)
        ax.text(left + ntimes_texture[i]/2, y_pos,
                texture[i], ha='center', va='center', fontsize=fontsize - 2)
        left += ntimes_texture[i]

    # ax.set_title('How fast do you want to go today?')

    # Texture

    # plt.show()

    output = performance_json.replace('.json', '.tex')
    tikz_save(output)


def paracheck(parameters):
    output = dict()
    # print parameters

    f = parameters['semantic_features']
    total = float(len(f))
    count_semantic = sum([i == 'True' for i in f])
    ratio_semantic = count_semantic/total
    print("Semantic: " + str(ratio_semantic))
    output['semantic_features'] = ratio_semantic

    f = parameters['patient_features']
    count_patient = sum([i == 'True' for i in f])
    ratio_patient = count_patient/total
    print("patient: " + str(ratio_patient))
    output['patient_features'] = ratio_patient

    f = parameters['orientation_features']
    count_orientation = sum([i == 'True' for i in f])
    ratio_orientation = count_orientation/total
    print("orientation: " + str(ratio_orientation))
    output['orientation_features'] = ratio_orientation

    f = parameters['histogram_features']
    count_histogram = sum([i == 'True' for i in f])
    ratio_histogram = count_histogram/total
    print("histogram: " + str(ratio_histogram))
    output['histogram_features'] = ratio_histogram

    f = parameters['shape_features']
    count_shape = sum([i == 'True' for i in f])
    ratio_shape = count_shape/total
    print("shape: " + str(ratio_shape))
    output['shape_features'] = ratio_shape

    if 'coliage_features' in parameters.keys():
        f = parameters['coliage_features']
        count_coliage = sum([i == 'True' for i in f])
        ratio_coliage = count_coliage/total
        print("coliage: " + str(ratio_coliage))
        output['coliage_features'] = ratio_coliage

    if 'phase_features' in parameters.keys():
        f = parameters['phase_features']
        count_phase = sum([i == 'True' for i in f])
        ratio_phase = count_phase/total
        print("phase: " + str(ratio_phase))
        output['phase_features'] = ratio_phase

    if 'vessel_features' in parameters.keys():
        f = parameters['vessel_features']
        count_vessel = sum([i == 'True' for i in f])
        ratio_vessel = count_vessel/total
        print("vessel: " + str(ratio_vessel))
        output['vessel_features'] = ratio_vessel

    if 'log_features' in parameters.keys():
        f = parameters['log_features']
        count_log = sum([i == 'True' for i in f])
        ratio_log = count_log/total
        print("log: " + str(ratio_log))
        output['log_features'] = ratio_log

    f = parameters['texture_features']
    count_texture_all = sum([i == 'True' for i in f])
    ratio_texture_all = count_texture_all/total
    print("texture_all: " + str(ratio_texture_all))
    output['texture_all_features'] = ratio_texture_all

    count_texture_no = sum([i == 'False' for i in f])
    ratio_texture_no = count_texture_no/total
    print("texture_no: " + str(ratio_texture_no))
    output['texture_no_features'] = ratio_texture_no

    count_texture_Gabor = sum([i == 'Gabor' for i in f])
    ratio_texture_Gabor = count_texture_Gabor/total
    print("texture_Gabor: " + str(ratio_texture_Gabor))
    output['texture_Gabor_features'] = ratio_texture_Gabor

    count_texture_LBP = sum([i == 'LBP' for i in f])
    ratio_texture_LBP = count_texture_LBP/total
    print("texture_LBP: " + str(ratio_texture_LBP))
    output['texture_LBP_features'] = ratio_texture_LBP

    count_texture_GLCM = sum([i == 'GLCM' for i in f])
    ratio_texture_GLCM = count_texture_GLCM/total
    print("texture_GLCM: " + str(ratio_texture_GLCM))
    output['texture_GLCM_features'] = ratio_texture_GLCM

    count_texture_GLRLM = sum([i == 'GLRLM' for i in f])
    ratio_texture_GLRLM = count_texture_GLRLM/total
    print("texture_GLRLM: " + str(ratio_texture_GLRLM))
    output['texture_GLRLM_features'] = ratio_texture_GLRLM

    count_texture_GLSZM = sum([i == 'GLSZM' for i in f])
    ratio_texture_GLSZM = count_texture_GLSZM/total
    print("texture_GLSZM: " + str(ratio_texture_GLSZM))
    output['texture_GLSZM_features'] = ratio_texture_GLSZM

    f = parameters['degree']
    print("Polynomial Degree: " + str(np.mean(f)))
    output['polynomial_degree'] = np.mean(f)

    return output


def main():
    if len(sys.argv) == 1:
        # print("TODO: Put in an example")
        performance_json = '/home/martijn/git/mets-paper-code/results/martijn_mskcc100_ensemble50_performance.json'

    elif len(sys.argv) != 2:
        raise IOError("This function accepts two arguments")
    else:
        performance_json = sys.argv[1]
    barplot(performance_json)


if __name__ == '__main__':
    main()
