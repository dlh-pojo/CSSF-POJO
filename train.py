
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def train_transform2():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/media/nc438/44C46749C4673BF4/wzq/IEcontrast8/data/airfield2/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/media/nc438/44C46749C4673BF4/wzq/IEcontrast8/data/airfield/',
                    help='Directory path to a batch of style images')
parser.add_argument('--edge_dir', type=str, default='/media/nc438/44C46749C4673BF4/wzq/IEcontrast8/data/airfield3',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples5', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=150000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=2.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--contrastive_weight_c', type=float, default=0.3)
parser.add_argument('--contrastive_weight_s', type=float, default=0.3)
parser.add_argument('--gan_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
parser.add_argument('--skip_connection_3', action='store_true',help='if specified, add skip connection on ReLU-3')
parser.add_argument('--shallow_layer', action='store_true',
                    help='if specified, also use features of shallow layers')
parser.add_argument('--l_nrom', type=float, default=100,
                    help='')
parser.add_argument('--l_cent', type=float, default=50,
                    help='')
parser.add_argument('--ab_nrom', type=float, default=110,
                    help='')
args = parser.parse_args('')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device('cuda:0')
#多卡运行请解除注释，并注释下一行
device = torch.device('cuda')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.Decoder(True)
vgg = net.vgg

valid = 1
fake = 0
D = net.MultiDiscriminator()
D.to(device)

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter, True, True, device)
# network = nn.DataParallel(network,device_ids=[1])
# 多卡操作时，请解除注释
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()
edge_tf = train_transform()


content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)


content_iter = iter(data.DataLoader(
    content_dataset, batch_size=int(args.batch_size ),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=int(args.batch_size ),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
# edge_iter = iter(data.DataLoader(
#     edge_dataset, batch_size=int(args.batch_size / 2),
#     sampler=InfiniteSamplerWrapper(edge_dataset),
#     num_workers=args.n_threads))

# optimizer = torch.optim.Adam([{'params': network.module.decoder.parameters()},
#                               {'params': network.module.transform.parameters()},
#                               {'params': network.module.proj_style.parameters()},
#                               {'params': network.module.net_adaattn_3.parameters()},
#                               {'params': network.module.proj_content.parameters()}], lr=args.lr)
# 多卡操作时请解除注释，同时注释掉下方的代码
                              
optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()},
                              {'params': network.proj_style.parameters()},
                              {'params': network.net_adaattn_3.parameters()},
                              {'params': network.proj_content.parameters()}], lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('./experiments/optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i)
    content_images = next(content_iter).to(device)
    content_images = rgb2lab(content_images)
    style_images = next(style_iter).to(device)
    # edge_images = next(edge_iter).to(device)
    style_images = rgb2lab(style_images)
    ######################################################
    # content_images_ = content_images[1:]
    # content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
    # content_images = torch.cat([content_images, content_images_], 0)
    # style_images = torch.cat([style_images, style_images], 0)
    # edge_images = torch.cat([edge_images, edge_images], 0)

    ######################################################

    img, loss_c, loss_s, l_identity1, l_identity2  = network(content_images, style_images, args.batch_size, True, True)
    # img = lab2rgb(img, args)
    # train discriminator
    loss_gan_d = D.compute_loss(style_images, valid) + D.compute_loss(img.detach(), fake)
    optimizer_D.zero_grad()
    loss_gan_d.backward()
    optimizer_D.step()

    # train generator
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    # loss_contrastive_c = args.contrastive_weight_c * loss_contrastive_c
    # loss_contrastive_s = args.contrastive_weight_s * loss_contrastive_s
    loss_gan_g = args.gan_weight * D.compute_loss(img, valid)
    loss = loss_s + loss_c + l_identity2 * 1 + l_identity1 * 50  + loss_gan_g 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    # writer.add_scalar('loss_contrastive_c', loss_contrastive_c.item(), i + 1)  # attention
    # writer.add_scalar('loss_contrastive_s', loss_contrastive_s.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)
    # writer.add_scalar('loss_hist', loss_hist.item(), i + 1)
    writer.add_scalar('loss', loss.item(), i+1)

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i == 0) or ((i + 1) % 500 == 0) or (i==150000):
        # content_images = lab2rgb(content_images, args)
        # style_images = lab2rgb(style_images,args)
        # img = lab2rgb(img, args)
        g_t_lab = img
        content_lab = content_images
        g_t_lab_1 = torch.cat([content_lab[:, [0], :, :], g_t_lab[:, [1], :, :]], dim=1)
        g_t_lab_2 = torch.cat([g_t_lab_1, g_t_lab[:, [2], :, :]], dim=1)
        g_t = lab2rgb(g_t_lab_2)
        content_images_rgb = lab2rgb(content_images)
        style_images_rgb = lab2rgb(style_images)
        output = torch.cat([style_images_rgb, content_images_rgb, g_t], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(output, str(output_name), nrow = args.batch_size)
    ##############################################cond##############################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter or (i==1):
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/network_iter_{:d}.pth'.format(args.save_dir,
                                                             i + 1))
writer.close()
