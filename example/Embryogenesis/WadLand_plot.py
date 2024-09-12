# -*- coding = utf-8 -*-
# @Time : 2024/8/6 15:07
# @Author : Yourui Han
# @File : WadLand_plot.py
# @Software : PyCharm


from utility import *


def plot_3d(func, data_train, train_time, integral_time, args, device):
    viz_samples = 20
    sigma_a = 0.01  # 0.01
    t_list = []
    z_s_samples = []
    z_s_data = []
    v = []
    g = []
    t_list2 = []
    odeint_setp = 1.0
    integral_time2 = np.arange(integral_time[0], integral_time[-1] + odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals=2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time), integral_time))[0]
    sample_time = list(reversed(sample_time))

    with torch.no_grad():
        for i in range(len(integral_time)):
            z_s0 = data_train[i]

            z_s_data.append(z_s0.cpu().detach().numpy())
            t_list2.append(integral_time[i])

        # traj backward
        z_s0 = Sampling_noise(viz_samples, train_time, len(train_time) - 1, data_train, sigma_a, device)
        p_diff_s0 = torch.zeros(z_s0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_s0.shape[0], 1).type(torch.float32).to(device)
        v_s = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_s0, g0, p_diff_s0))[0]
        g_s = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device), (z_s0, g0, p_diff_s0))[1]

        v.append(v_s.cpu().detach().numpy())
        g.append(g_s.cpu().detach().numpy())
        z_s_samples.append(z_s0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})

        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval': plot_time})
        z_s1, _, p_diff_s1 = odesolve(func, y0=(z_s0, g0, p_diff_s0), options=options)
        for i in range(len(plot_time) - 1):
            v_s = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_s1[i + 1], g0, p_diff_s1))[
                0]
            g_s = func(torch.tensor(plot_time[i + 1]).type(torch.float32).to(device), (z_s1[i + 1], g0, p_diff_s1))[
                1]

            z_s_samples.append(z_s1[i + 1].cpu().detach().numpy())
            g.append(g_s.cpu().detach().numpy())
            v.append(v_s.cpu().detach().numpy())
            t_list.append(plot_time[i + 1])

        aa = 5  # 3
        widths = 0.005  # arrow width
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 5

        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)

        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('tsne1')
        ax1.set_ylabel('tsne2')
        line_width = 0.4

        color_wanted = [np.array([250, 187, 110]) / 255,
                        np.array([173, 219, 136]) / 255,
                        np.array([250, 199, 179]) / 255,
                        np.array([238, 68, 49]) / 255,
                        np.array([206, 223, 239]) / 255,
                        np.array([3, 149, 198]) / 255,
                        np.array([180, 180, 213]) / 255,
                        np.array([178, 143, 237]) / 255]
        # not_v_list = [0, 2, 3, 4, 6, 9, 12, 15]
        not_v_list = []
        for j in range(viz_samples):  # individual traj
            if j not in not_v_list:
                for i in range(len(plot_time) - 1):
                    ax1.plot([z_s_samples[i][j, 0], z_s_samples[i + 1][j, 0]],
                             [z_s_samples[i][j, 1], z_s_samples[i + 1][j, 1]],
                             linewidth=0.5, color='grey', zorder=2)
        for i in range(len(integral_time)):
            ax1.scatter(z_s_data[i][:, 0], z_s_data[i][:, 1], s=aa * 10, linewidth=line_width,
                        alpha=1, facecolors='none', edgecolors=color_wanted[i], zorder=1)

        # add inferrred trajecotry
        for i in range(len(sample_time)):
            for j in range(z_s_samples[sample_time[i]].shape[0]):
                if j not in not_v_list:
                    ax1.scatter(z_s_samples[sample_time[i]][j, 0], z_s_samples[sample_time[i]][j, 1],
                                s=aa * 10, linewidth=0, color=color_wanted[i], zorder=3)
                    # ax1.quiver(z_s_samples[sample_time[i]][j, 0], z_s_samples[sample_time[i]][j, 1],
                    #            v[sample_time[i]][j, 0] / v_scale, v[sample_time[i]][j, 1] / v_scale,
                    #            color='k', alpha=0.8, linewidths=widths, zorder=4)
                    arrow_start = (z_s_samples[sample_time[i]][j, 0], z_s_samples[sample_time[i]][j, 1])
                    arrow_end = (z_s_samples[sample_time[i]][j, 0] + v[sample_time[i]][j, 0] / v_scale,
                                 z_s_samples[sample_time[i]][j, 1] + v[sample_time[i]][j, 1] / v_scale)

                    # 创建一个箭头对象
                    arrow = patches.FancyArrow(arrow_start[0], arrow_start[1], v[sample_time[i]][j, 0] / v_scale,
                                               v[sample_time[i]][j, 1] / v_scale,
                                               width=widths, color='k', head_width=0.015, head_length=0.02,
                                               shape='full', overhang=0.005, head_starts_at_zero=False)
                    ax1.add_patch(arrow)
        # 修改X轴和Y轴的标签字体大小和加粗
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(12)
            item.set_fontweight('bold')

        plt.show()

        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('pca1')
        ax1.set_ylabel('pca2')
        for i in range(len(integral_time)):
            ax1.scatter(z_s_data[i][:, 0], z_s_data[i][:, 1], s=aa * 10, linewidth=line_width,
                        alpha=0.8, facecolors='none', edgecolors=color_wanted[i], zorder=1)
        # add inferrred cell center trajecotry
        center_x = []
        center_y = []
        for i in range(len(integral_time)):
            center_x.append(np.mean(z_s_samples[i][:, 0], axis=0))
            center_y.append(np.mean(z_s_samples[i][:, 1], axis=0))
        for i in range(len(integral_time) - 1):
            ax1.plot([center_x[i], center_x[i + 1]], [center_y[i], center_y[i + 1]], linewidth=line_width * 5,
                     linestyle='dashed', color='black', zorder=2)
        for i in range(len(integral_time)):
            ax1.scatter(center_x[i], center_y[i], s=aa * 40, linewidth=line_width, marker='*',
                        alpha=1, facecolors=color_wanted[len(integral_time) - i - 1],
                        edgecolors=color_wanted[len(integral_time) - i - 1], zorder=3)

        # add cell center trajectory
        center_x = []
        center_y = []
        for i in range(len(integral_time)):
            center_x.append(np.mean(z_s_data[i][:, 0], axis=0))
            center_y.append(np.mean(z_s_data[i][:, 1], axis=0))
        for i in range(len(integral_time) - 1):
            ax1.plot([center_x[i], center_x[i + 1]], [center_y[i], center_y[i + 1]], linewidth=line_width * 5,
                     linestyle='solid', color='grey', zorder=2)
        for i in range(len(integral_time)):
            ax1.scatter(center_x[i], center_y[i], s=aa * 20, linewidth=line_width, marker='o',
                        alpha=1, facecolors=color_wanted[i], edgecolors=color_wanted[i], zorder=3)
        plt.show()


if __name__ == '__main__':
    arguments = input_args()

    torch.enable_grad()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    device = torch.device('cuda:' + str(arguments.gpu)
                          if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(arguments, device)
    integral_time = arguments.pseudo_time

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i, x in enumerate(time_pts) if i != leave_1_out]

    # model
    func = RWL(in_out_dim=data_train[0].shape[1], GRN_dim_hidden=arguments.GRN_dim_hidden,
               GRN_num_hiddens=arguments.GRN_num_hiddens, BRD_dim_hidden=arguments.BRD_dim_hidden,
               BRD_num_hiddens=arguments.BRD_num_hiddens, activation=arguments.activation,
               decrease_multipleint=arguments.decrease_multipleint, sparsity_param=arguments.sparsity_param).to(device)

    if arguments.save_dir is not None:
        if not os.path.exists(arguments.save_dir):
            os.makedirs(arguments.save_dir)
        ckpt_path = os.path.join(arguments.save_dir, 'ckpt_itr6000.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            func.load_state_dict(checkpoint['func_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    # generate the plot of trajecotry
    plot_3d(func, data_train, train_time, integral_time, arguments, device)

    # energy
    cell_type = []
    all_type = ['Zygote', '2-cell', '4-cell', '8-cell', 'Morula']
    colors = [np.array([250, 187, 110]) / 255,
              np.array([173, 219, 136]) / 255,
              np.array([250, 199, 179]) / 255,
              np.array([238, 68, 49]) / 255,
              np.array([206, 223, 239]) / 255]
    for i in range(len(integral_time)):
        for j in range(data_train[i].shape[0]):
            cell_type.append(all_type[i])
    value = []
    energy = []
    cost = []
    g = []

    energy_np = []
    for i in range(len(integral_time)):
        z_s = data_train[i]
        g_s0 = torch.zeros(1, 1).type(torch.float32).to(device)
        p_diff_s0 = torch.ones(z_s.shape[0], 1).type(torch.float32).to(device)
        g_s = func(torch.tensor(i).type(torch.float32).to(device), (z_s, g_s0, p_diff_s0))[1]
        g.append(g_s.cpu().detach().numpy())

    for i in range(len(integral_time)):
        z_s = data_train[i]
        g_s0 = torch.zeros(1, 1).type(torch.float32).to(device)
        p_diff_s0 = torch.ones(z_s.shape[0], 1).type(torch.float32).to(device)
        # compute the mean of jacobian of v within cells z_t at time (time_pt)
        dim = z_s.shape[1]
        energy_list = []
        for j in range(z_s.shape[0]):
            x_s = z_s[j, :].reshape([1, dim])
            v_xs = func(torch.tensor(i).type(torch.float32).to(device), (x_s, g_s0, p_diff_s0))[0]
            jac = Jacobian(v_xs, x_s).reshape(dim, dim).detach().cpu().numpy()
            GRN = jac[:dim - 1, :dim - 1]
            x_s = z_s[j, :dim - 1].reshape([1, dim - 1]).detach().cpu().numpy()
            x_s_t = z_s[j, :dim - 1].reshape([dim - 1, 1]).detach().cpu().numpy()
            energy.append((-x_s @ GRN @ x_s_t)[0][0])
            cost.append(-g[i][j][0] * np.sqrt((x_s_t @ x_s)[0][0]))  #
            energy_list.append((-x_s @ GRN @ x_s_t)[0][0] - g[i][j][0] * np.sqrt((x_s_t @ x_s)[0][0]))
        energy_np.append(energy_list)
    data = pd.DataFrame({
        'Category': cell_type,
        'The potential of cell migration': energy
    })
    sns.violinplot(x="Category", y="The potential of cell migration", data=data, palette=colors)
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(12)  # 设置字体大小
        item.set_fontweight('bold')  # 设置字体加粗
    plt.show()

    data = pd.DataFrame({
        'Category': cell_type,
        'The potential of cell proliferation': cost
    })
    sns.violinplot(x="Category", y="The potential of cell proliferation", data=data, palette=colors)
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(12)  # 设置字体大小
        item.set_fontweight('bold')  # 设置字体加粗

    plt.show()

    value = [a + b for a, b in zip(energy, cost)]
    data = pd.DataFrame({
        'Category': cell_type,
        'Cell potential': value
    })
    sns.violinplot(x="Category", y="Cell potential", data=data, palette=colors)

    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(12)  # 设置字体大小
        item.set_fontweight('bold')  # 设置字体加粗
    plt.ylim(-3, 1)
    plt.show()

    ax1 = plt.axes(projection='3d')
    ax1.grid(False)
    ax1.set_xlabel('tsne1')
    ax1.set_ylabel('tsne2')
    ax1.set_zlabel('Cell potential')
    line_width = 0.4
    z_s_data = []
    for i in range(len(integral_time)):
        z_s0 = data_train[i]
        z_s_data.append(z_s0.cpu().detach().numpy())
    for i in range(len(integral_time)):
        ax1.scatter(z_s_data[i][:, 0], z_s_data[i][:, 1], energy_np[i],
                    s=5 * 10, alpha=0.2, linewidth=0, facecolors=colors[i], edgecolors=colors[i], zorder=1)
    # add inferrred cell center trajecotry
    center_x = []
    center_y = []
    center_z = []
    for i in range(len(integral_time)):
        center_x.append(np.mean(z_s_data[i][:, 0], axis=0))
        center_y.append(np.mean(z_s_data[i][:, 1], axis=0))
        center_z.append(sum(energy_np[i]) / len(energy_np[i]))
    for i in range(len(integral_time) - 1):
        ax1.plot([center_x[i], center_x[i + 1]], [center_y[i], center_y[i + 1]], [center_z[i], center_z[i + 1]],
                 linewidth=line_width * 5,
                 linestyle='dashed', color='black', zorder=2)
    for i in range(len(integral_time)):
        ax1.scatter(center_x[i], center_y[i], center_z[i], s=5 * 20, linewidth=line_width, marker='*',
                    alpha=1, facecolors='black',
                    edgecolors='black', zorder=3)
    # 设置坐标轴框体颜色（可选，默认为灰色）
    ax1.xaxis.pane_color = (1.0, 1.0, 1.0, 0.0)  # x轴面板颜色为白色（最后一个值为alpha）
    ax1.yaxis.pane_color = (1.0, 1.0, 1.0, 0.0)  # y轴面板颜色为白色
    ax1.zaxis.pane_color = (1.0, 1.0, 1.0, 0.0)  # z轴面板颜色为白色
    for item in ([ax1.xaxis.label, ax1.yaxis.label, ax1.zaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels()):
        item.set_fontsize(12)  # 设置字体大小
        item.set_fontweight('bold')  # 设置字体加粗
    ax1.set_zlim(-3, 1)
    plt.show()
