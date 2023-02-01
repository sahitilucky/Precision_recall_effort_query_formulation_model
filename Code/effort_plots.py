import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(methods, filename, method_labels):
    cm = plt.get_cmap('tab20')
    NUM_COLORS = len(methods)
    markers = ['o', '^', '+', 's', 'D', '*', 'x', 'v']
    plt.clf()
    plt.cla()
    count_colors = 0
    for idx in range(len(methods)):
        method = methods[idx][:]
        label = method_labels[idx]
        plt.plot([l[0] for l in method], [float(l[1])/float(4) for l in method], label=label, color=cm(1. * count_colors / NUM_COLORS),
                 marker=markers[count_colors])
        # print (label, label.split("test"),test_k)
        # print (label.split("test" + test_k)[1])

        # +label.split('test')[0] if args.model.lower().strip() == 'lambda' else args.model
        # print (label)
        count_colors = count_colors + 1
    plt.xlabel('alpha parameter')
    plt.ylabel('Jaccard similarity')
    plt.ylim([0.1, 0.5])
    plt.xlim([0.0, 1.0])
    plt.title('Jaccard similarity-alpha')
    plt.legend(loc="best")
    plt.savefig(filename, facecolor='white',
                edgecolor='none', bbox_inches="tight")


#Session track 2014
Cr_R12g_Cp_add = [(0, 2.0866666666666664), (1, 0.2776346801346802), (2, 0.41207076719576724), (3, 0.42880170755170754), (4, 0.4336541606541606), (5, 0.44556731000481), (6, 0.44308849021349017)]
Cr_R12g_Dp_add = [(0, 2.0866666666666664), (1, 0.2776346801346802), (2, 0.4077374338624339), (3, 0.4359842472342472), (4, 0.4359842472342472), (5, 0.4359842472342472), (6, 0.4359842472342472)]
Dr_R12g_Cp_add = [(0, 5.333333333333333), (1, 0.2776346801346802), (2, 0.42376521164021164), (3, 0.43545448532948533), (4, 0.43876328763828765), (5, 0.450676436988937), (6, 0.4504433020683021)]
Dr_R12g_Dp_add = [(0, 1.8766666666666667), (1, 0.2776346801346802), (2, 0.41965410052910046), (3, 0.44359535834535835), (4, 0.44359535834535835), (5, 0.44359535834535835), (6, 0.44359535834535835)]
#Cr = [(1, 0.2530575292260075), (2, 0.3766973199907982), (3, 0.4124071669071669), (4, 0.4124071669071669), (5, 0.4124071669071669), (6, 0.4124071669071669)]
#Dr = [(1, 0.2530575292260075), (2, 0.3762112088796871), (3, 0.3987480054164836), (4, 0.3987480054164836), (5, 0.40045633874981695), (6, 0.40451387843235664)]
#Cp = [(1, 0.2500726888851889), (2, 0.3043870504495505), (3, 0.2980340284715285), (4, 0.3305023841898842), (5, 0.36411231823731827), (6, 0.3573099377474377)]
#Dp = [(1, 0.2500726888851889), (2, 0.2985902454027454), (3, 0.3129973221223221), (4, 0.3129973221223221), (5, 0.3129973221223221), (6, 0.3129973221223221)]
dataset = "Session_track_2014_effort"
method_labels = ["CrCp", "CrDp", "DrCp", "DrDp"]
methods = [Cr_R12g_Cp_add, Cr_R12g_Dp_add, Dr_R12g_Cp_add, Dr_R12g_Dp_add]
#plot(methods, "../figures/" + dataset + ".jpg", method_labels)

Cr_R12g_Cp_add1 = [(0.0, 0.27341629666629674), (0.1, 0.33378418803418797), (0.2, 0.33214224664224656), (0.3, 0.3296581196581197), (0.4, 0.3311095848595849), (0.5, 0.3319398656898657), (0.6, 0.3296581196581197), (0.7, 0.3296581196581197), (0.8, 0.3296581196581197), (0.9, 0.3296581196581197), (1.0, 0.30381216931216926)]
Cr_R12g_Cp_add2 = [(0.0, 0.28311159211159215), (0.1, 0.3509550264550265), (0.2, 0.3510952380952382), (0.3, 0.3489246031746032), (0.4, 0.3478002645502646), (0.5, 0.350234126984127), (0.6, 0.3489246031746032), (0.7, 0.3489246031746032), (0.8, 0.3489246031746032), (0.9, 0.3489246031746032), (1.0, 0.3253068783068783)]
Cr_R12g_Cp_add3 =  [(0.0, 0.27413323713323706), (0.1, 0.31696957671957665), (0.2, 0.3148465608465608), (0.3, 0.31314177489177486), (0.4, 0.3149473304473304), (0.5, 0.31399603174603175), (0.6, 0.31314177489177486), (0.7, 0.31314177489177486), (0.8, 0.31314177489177486), (0.9, 0.31314177489177486), (1.0, 0.29065476190476197)]
Cr_R12g_Cp_add4 =   [(0.0, 0.2978694083694083), (0.1, 0.3638934583934584), (0.2, 0.36354058904058906), (0.3, 0.36036959336959334), (0.4, 0.36199230399230403), (0.5, 0.36261165686165686), (0.6, 0.36036959336959334), (0.7, 0.36036959336959334), (0.8, 0.36036959336959334), (0.9, 0.36036959336959334), (1.0, 0.3352686387686387)]

Cr_R12g_Cp_add = Cr_R12g_Cp_add1
for idx,instance in enumerate(Cr_R12g_Cp_add2):
    Cr_R12g_Cp_add[idx] = (Cr_R12g_Cp_add[idx][0], Cr_R12g_Cp_add[idx][1]+Cr_R12g_Cp_add2[idx][1])
for idx,instance in enumerate(Cr_R12g_Cp_add3):
    Cr_R12g_Cp_add[idx] = (Cr_R12g_Cp_add[idx][0], Cr_R12g_Cp_add[idx][1]+Cr_R12g_Cp_add3[idx][1])
for idx,instance in enumerate(Cr_R12g_Cp_add4):
    Cr_R12g_Cp_add[idx] = (Cr_R12g_Cp_add[idx][0], Cr_R12g_Cp_add[idx][1]+Cr_R12g_Cp_add4[idx][1])

Cr_R12g_Dp_add1 =  [(0.0, 0.2214062049062049), (0.1, 0.32658730158730154), (0.2, 0.32658730158730154), (0.3, 0.32658730158730154), (0.4, 0.32658730158730154), (0.5, 0.32658730158730154), (0.6, 0.32658730158730154), (0.7, 0.32658730158730154), (0.8, 0.32658730158730154), (0.9, 0.32658730158730154), (1.0, 0.30381216931216926)]
Cr_R12g_Dp_add2 =  [(0.0, 0.23337169312169312), (0.1, 0.3445978835978837), (0.2, 0.3445978835978837), (0.3, 0.3445978835978837), (0.4, 0.3445978835978837), (0.5, 0.3445978835978837), (0.6, 0.3445978835978837), (0.7, 0.3445978835978837), (0.8, 0.3445978835978837), (0.9, 0.3445978835978837), (1.0, 0.3253068783068783)]
Cr_R12g_Dp_add3 = [(0.0, 0.2308451548451549), (0.1, 0.3179775132275131), (0.2, 0.3179775132275131), (0.3, 0.3179775132275131), (0.4, 0.3179775132275131), (0.5, 0.3179775132275131), (0.6, 0.3179775132275131), (0.7, 0.3179775132275131), (0.8, 0.3179775132275131), (0.9, 0.3179775132275131), (1.0, 0.29065476190476197)]
Cr_R12g_Dp_add4=  [(0.0, 0.2205), (0.1, 0.35400024050024037), (0.2, 0.35400024050024037), (0.3, 0.35400024050024037), (0.4, 0.35400024050024037), (0.5, 0.35400024050024037), (0.6, 0.35400024050024037), (0.7, 0.35400024050024037), (0.8, 0.35400024050024037), (0.9, 0.35400024050024037), (1.0, 0.3352686387686387)]

Cr_R12g_Dp_add = Cr_R12g_Dp_add1
for idx, instance in enumerate(Cr_R12g_Dp_add2):
    Cr_R12g_Dp_add[idx] = (Cr_R12g_Dp_add[idx][0], Cr_R12g_Dp_add[idx][1] + Cr_R12g_Dp_add2[idx][1])
for idx, instance in enumerate(Cr_R12g_Dp_add3):
    Cr_R12g_Dp_add[idx] = (Cr_R12g_Dp_add[idx][0], Cr_R12g_Dp_add[idx][1] + Cr_R12g_Dp_add3[idx][1])
for idx, instance in enumerate(Cr_R12g_Dp_add4):
    Cr_R12g_Dp_add[idx] = (Cr_R12g_Dp_add[idx][0], Cr_R12g_Dp_add[idx][1] + Cr_R12g_Dp_add4[idx][1])

Dr_R12g_Cp_add1 = [(0.0, 0.27341629666629674), (0.1, 0.341300061050061), (0.2, 0.34018894993894994), (0.3, 0.341300061050061), (0.4, 0.34018894993894994), (0.5, 0.34200532800532796), (0.6, 0.3420266955266955), (0.7, 0.3420266955266955), (0.8, 0.3409420394420393), (0.9, 0.33938359788359784), (1.0, 0.29924867724867726)]
Dr_R12g_Cp_add2 = [(0.0, 0.28311159211159215), (0.1, 0.3573597883597884), (0.2, 0.3556931216931217), (0.3, 0.3573597883597884), (0.4, 0.3556931216931217), (0.5, 0.35806613756613764), (0.6, 0.35806613756613764), (0.7, 0.35806613756613764), (0.8, 0.3587328042328043), (0.9, 0.35710582010582015), (1.0, 0.32127455716586156)]
Dr_R12g_Cp_add3 = [(0.0, 0.27413323713323706), (0.1, 0.3226044973544973), (0.2, 0.3218108465608465), (0.3, 0.3225251322751322), (0.4, 0.3218108465608465), (0.5, 0.32321296296296287), (0.6, 0.32321296296296287), (0.7, 0.323292328042328), (0.8, 0.32321296296296287), (0.9, 0.3207526455026455), (1.0, 0.28452910052910046)]
Dr_R12g_Cp_add4 = [(0.0, 0.2978694083694083), (0.1, 0.37105218855218863), (0.2, 0.36938552188552193), (0.3, 0.3706553631553632), (0.4, 0.36938552188552193), (0.5, 0.3721712361712362), (0.6, 0.3721712361712362), (0.7, 0.37256806156806166), (0.8, 0.3705045695045695), (0.9, 0.3661990139490141), (1.0, 0.32595646945646933)]

Dr_R12g_Cp_add = Dr_R12g_Cp_add1
for idx, instance in enumerate(Dr_R12g_Cp_add2):
    Dr_R12g_Cp_add[idx] = (Dr_R12g_Cp_add[idx][0], Dr_R12g_Cp_add[idx][1] + Dr_R12g_Cp_add2[idx][1])
for idx, instance in enumerate(Dr_R12g_Cp_add3):
    Dr_R12g_Cp_add[idx] = (Dr_R12g_Cp_add[idx][0], Dr_R12g_Cp_add[idx][1] + Dr_R12g_Cp_add3[idx][1])
for idx, instance in enumerate(Dr_R12g_Cp_add4):
    Dr_R12g_Cp_add[idx] = (Dr_R12g_Cp_add[idx][0], Dr_R12g_Cp_add[idx][1] + Dr_R12g_Cp_add4[idx][1])


Dr_R12g_Dp_add1 = [(0.0, 0.2214062049062049), (0.1, 0.3323650793650793), (0.2, 0.3323650793650793), (0.3, 0.3323650793650793), (0.4, 0.3323650793650793), (0.5, 0.3323650793650793), (0.6, 0.3323650793650793), (0.7, 0.3323650793650793), (0.8, 0.3323650793650793), (0.9, 0.3323650793650793), (1.0, 0.29924867724867726)]
Dr_R12g_Dp_add2 =  [(0.0, 0.23337169312169312), (0.1, 0.350931216931217), (0.2, 0.350931216931217), (0.3, 0.350931216931217), (0.4, 0.350931216931217), (0.5, 0.350931216931217), (0.6, 0.350931216931217), (0.7, 0.350931216931217), (0.8, 0.350931216931217), (0.9, 0.350931216931217), (1.0, 0.32127455716586156)]
Dr_R12g_Dp_add3 =   [(0.0, 0.2308451548451549), (0.1, 0.32431084656084647), (0.2, 0.32431084656084647), (0.3, 0.32431084656084647), (0.4, 0.32431084656084647), (0.5, 0.32431084656084647), (0.6, 0.32431084656084647), (0.7, 0.32431084656084647), (0.8, 0.32431084656084647), (0.9, 0.32431084656084647), (1.0, 0.28452910052910046)]
Dr_R12g_Dp_add4 =   [(0.0, 0.2205), (0.1, 0.35861135161135155), (0.2, 0.35861135161135155), (0.3, 0.35861135161135155), (0.4, 0.35861135161135155), (0.5, 0.35861135161135155), (0.6, 0.35861135161135155), (0.7, 0.35861135161135155), (0.8, 0.35861135161135155), (0.9, 0.35861135161135155), (1.0, 0.32595646945646933)]

Dr_R12g_Dp_add = Dr_R12g_Dp_add1
for idx, instance in enumerate(Dr_R12g_Dp_add2):
    Dr_R12g_Dp_add[idx] = (Dr_R12g_Dp_add[idx][0], Dr_R12g_Dp_add[idx][1] + Dr_R12g_Dp_add2[idx][1])
for idx, instance in enumerate(Dr_R12g_Dp_add3):
    Dr_R12g_Dp_add[idx] = (Dr_R12g_Dp_add[idx][0], Dr_R12g_Dp_add[idx][1] + Dr_R12g_Dp_add3[idx][1])
for idx, instance in enumerate(Dr_R12g_Dp_add4):
    Dr_R12g_Dp_add[idx] = (Dr_R12g_Dp_add[idx][0], Dr_R12g_Dp_add[idx][1] + Dr_R12g_Dp_add4[idx][1])

dataset = "Session_track_2014_sensitivity"
method_labels = ["CrCp", "CrDp", "DrCp", "DrDp"]
methods = [Cr_R12g_Cp_add, Cr_R12g_Dp_add, Dr_R12g_Cp_add, Dr_R12g_Dp_add]
plot(methods, "../figures/" + dataset + ".jpg", method_labels)

'''
#Session track 2013
Cr_R12g_Cp_add [(0, 2.1510204081632653), (1, 0.16782586248785228), (2, 0.31487454108627566), (3, 0.3491312020840081), (4, 0.35691620100421123), (5, 0.36452124500593885), (6, 0.36116587911018183)]
Cr_R12g_Dp_add [(0, 2.1510204081632653), (1, 0.16782586248785228), (2, 0.31480931459885536), (3, 0.3539317770759097), (4, 0.3539317770759097), (5, 0.3539317770759097), (6, 0.3539317770759097)]
Dr_R12g_Cp_add [(0, 5.363265306122449), (1, 0.16782586248785228), (2, 0.3187744304070834), (3, 0.3543297497570457), (4, 0.3599735787172012), (5, 0.3675786227189288), (6, 0.3650929811559063)]
Dr_R12g_Dp_add [(0, 1.8489795918367347), (1, 0.16782586248785228), (2, 0.3200462962962963), (3, 0.3533347843105496), (4, 0.3533347843105496), (5, 0.3533347843105496), (6, 0.3533347843105496)]
Cr [(1, 0.14925942797754022), (2, 0.2935732844725191), (3, 0.3435404856387), (4, 0.3435404856387), (5, 0.3435404856387), (6, 0.3435404856387)]
Dr [(1, 0.14925942797754022), (2, 0.2941563748515279), (3, 0.3257528547133138), (4, 0.3257528547133138), (5, 0.3257528547133138), (6, 0.3312029460052418)]
Cp [(1, 0.1096232520786092), (2, 0.13850115178346473), (3, 0.12715457743944136), (4, 0.16332508249812672), (5, 0.21485642585557552), (6, 0.21079130714334793)]
Dp [(1, 0.1096232520786092), (2, 0.13774815084764064), (3, 0.14526540510384045), (4, 0.14526540510384045), (5, 0.14526540510384045), (6, 0.14526540510384045)]

#Session track 2012
Cr_R12g_Cp_add [(0, 2.004166666666667), (1, 0.1345626653439153), (2, 0.2478695436507936), (3, 0.25546784812409806), (4, 0.247365395021645), (5, 0.25071499433106575), (6, 0.2536001275510204)]
Cr_R12g_Dp_add [(0, 2.004166666666667), (1, 0.1345626653439153), (2, 0.24941468253968252), (3, 0.26697172619047616), (4, 0.26697172619047616), (5, 0.26697172619047616), (6, 0.26697172619047616)]
Dr_R12g_Cp_add [(0, 5.2625), (1, 0.1345626653439153), (2, 0.253531746031746), (3, 0.2587118957431457), (4, 0.2481590458152958), (5, 0.25150864512471655), (6, 0.2527611252834467)]
Dr_R12g_Dp_add [(0, 1.8791666666666667), (1, 0.1345626653439153), (2, 0.25445932539682536), (3, 0.2730059523809524), (4, 0.2730059523809524), (5, 0.2730059523809524), (6, 0.2730059523809524)]
Cr [(1, 0.13189484126984125), (2, 0.23065476190476192), (3, 0.25289682539682534), (4, 0.25289682539682534), (5, 0.25289682539682534), (6, 0.25289682539682534)]
Dr [(1, 0.13189484126984125), (2, 0.23034226190476187), (3, 0.2407316468253968), (4, 0.2407316468253968), (5, 0.2407316468253968), (6, 0.2407316468253968)]
Cp [(1, 0.12257971938775508), (2, 0.19337797619047614), (3, 0.22984877473716753), (4, 0.24458323670377238), (5, 0.2499723639455782), (6, 0.2538013747165533)]
Dp [(1, 0.12257971938775508), (2, 0.18508680555555557), (3, 0.20377641207998348), (4, 0.20377641207998348), (5, 0.20377641207998348), (6, 0.20377641207998348)]

#Session track 2012
Cr_R12g_Cp_add [(0, 2.004166666666667), (1, 0.19769074409796167), (2, 0.3707204477745601), (3, 0.3869958485110402), (4, 0.38007915944674586), (5, 0.3844363987056906), (6, 0.3892698839643294)]
Cr_R12g_Dp_add [(0, 2.004166666666667), (1, 0.19769074409796167), (2, 0.3726964684302089), (3, 0.3991172051066188), (4, 0.3991172051066188), (5, 0.3991172051066188), (6, 0.3991172051066188)]
Dr_R12g_Cp_add [(0, 5.2625), (1, 0.19769074409796167), (2, 0.37795302745103365), (3, 0.3916912936367375), (4, 0.381304251187788), (5, 0.3974356362721325), (6, 0.3875685981299677)]
Dr_R12g_Dp_add [(0, 1.8791666666666667), (1, 0.19769074409796167), (2, 0.3789945803180472), (3, 0.4071246542237654), (4, 0.4071246542237654), (5, 0.4071246542237654), (6, 0.4071246542237654)]
Cr [(1, 0.19374508796707404), (2, 0.348347139418924), (3, 0.38549513793946866), (4, 0.38549513793946866), (5, 0.38549513793946866), (6, 0.38549513793946866)]
Dr [(1, 0.19374508796707404), (2, 0.347722831078752), (3, 0.3662911644444826), (4, 0.3662911644444826), (5, 0.3662911644444826), (6, 0.3662911644444826)]
Cp [(1, 0.1795528183929917), (2, 0.29189646306107275), (3, 0.35190955296787596), (4, 0.3730040137025694), (5, 0.37724786221868656), (6, 0.3835791135225654)]
Dp [(1, 0.1795528183929917), (2, 0.281850391841895), (3, 0.3160850071084475), (4, 0.3160850071084475), (5, 0.3160850071084475), (6, 0.3160850071084475)]

#Session track 2013
Cr_R12g_Cp_add [(0, 2.1510204081632653), (1, 0.24827802749807173), (2, 0.4634269580399541), (3, 0.5080914262181973), (4, 0.5219431166058808), (5, 0.5333650537824416), (6, 0.5323910070182785)]
Cr_R12g_Dp_add [(0, 2.1510204081632653), (1, 0.24827802749807173), (2, 0.4637125037445612), (3, 0.5169401631368683), (4, 0.5169401631368683), (5, 0.5169401631368683), (6, 0.5169401631368683)]
Dr_R12g_Cp_add [(0, 5.363265306122449), (1, 0.24827802749807173), (2, 0.46968357049854703), (3, 0.5155858104444959), (4, 0.5244388801592746), (5, 0.5357003510908502), (6, 0.5344150743833969)]
Dr_R12g_Dp_add [(0, 1.8489795918367347), (1, 0.24827802749807173), (2, 0.471807657727781), (3, 0.5166957550945579), (4, 0.5166957550945579), (5, 0.5166957550945579), (6, 0.5166957550945579)]
Cr [(1, 0.2201189786820607), (2, 0.42813583893624174), (3, 0.49576527011948235), (4, 0.49576527011948235), (5, 0.49576527011948235), (6, 0.49576527011948235)]
Dr [(1, 0.2201189786820607), (2, 0.4285881532672541), (3, 0.4703991005168377), (4, 0.4703991005168377), (5, 0.4703991005168377), (6, 0.4821993061894174)]
Cp [(1, 0.16301497670106582), (2, 0.2234538352561694), (3, 0.22026321228400753), (4, 0.2771095219832198), (5, 0.34940091324551803), (6, 0.3459190565909193)]
Dp [(1, 0.16301497670106582), (2, 0.2219824556225), (3, 0.24362380127417266), (4, 0.24362380127417266), (5, 0.24362380127417266), (6, 0.24362380127417266)]

#Session track 2014
Cr_R12g_Cp_add [(0, 2.0866666666666664), (1, 0.3908423443778506), (2, 0.5813729843694423), (3, 0.606285706331389), (4, 0.6160841871338021), (5, 0.639177417460075), (6, 0.6372810305120796)]
Cr_R12g_Dp_add [(0, 2.0866666666666664), (1, 0.3908423443778506), (2, 0.5733741500811004), (3, 0.6080517988360853), (4, 0.6080517988360853), (5, 0.6080517988360853), (6, 0.6080517988360853)]
Dr_R12g_Cp_add [(0, 5.333333333333333), (1, 0.3908423443778506), (2, 0.5934547998940086), (3, 0.6144066839082274), (4, 0.6239384864139237), (5, 0.6468120821297018), (6, 0.647408991147756)]
Dr_R12g_Dp_add [(0, 1.8766666666666667), (1, 0.3908423443778506), (2, 0.5868517296549058), (3, 0.6160279677475677), (4, 0.6160279677475677), (5, 0.6160279677475677), (6, 0.6160279677475677)]
Cr [(1, 0.3585466963931711), (2, 0.5362256589993373), (3, 0.5895447697224964), (4, 0.5895447697224964), (5, 0.5895447697224964), (6, 0.5895447697224964)]
Dr [(1, 0.3585466963931711), (2, 0.5359156364726265), (3, 0.5681697315112486), (4, 0.5681697315112486), (5, 0.5752590869187162), (6, 0.5854049969941026)]
Cp [(1, 0.34884074532260845), (2, 0.44641134225134915), (3, 0.46993129255895827), (4, 0.5112630797853501), (5, 0.5631010015770063), (6, 0.5568748776784828)]
Dp [(1, 0.34884074532260845), (2, 0.43902206700194835), (3, 0.47020941742856925), (4, 0.47020941742856925), (5, 0.47020941742856925), (6, 0.47020941742856925)]
'''


