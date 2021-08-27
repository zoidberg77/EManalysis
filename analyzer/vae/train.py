import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os
import glob
import datetime

from analyzer.data.data_vis import visptc
from analyzer.vae.model import unet
from analyzer.vae.model.ptc_vae import PTCvae
from chamferdist import ChamferDistance

from analyzer.vae.model.random_ptc_ae import RandomPtcDataModule
from analyzer.utils.vis.monitor import build_monitor


class Trainer:
	'''
	Unet training object.
	'''
	def __init__(self, dataset, train_percentage, optimizer_type, loss_function, cfg):
		self.cfg = cfg
		self.dataset = dataset
		self.model_type = cfg.AUTOENCODER.ARCHITECTURE
		self.epochs = cfg.AUTOENCODER.EPOCHS
		self.optimizer_type = optimizer_type
		self.device = 'cpu'
		if cfg.SYSTEM.NUM_GPUS > 0:
			self.device = 'cuda'
		self.loss_function = loss_function
		if self.model_type == "unet_3d":
			self.model = unet.UNet3D(input_shape=cfg.AUTOENCODER.TARGET, latent_space=cfg.AUTOENCODER.LATENT_SPACE)
		if self.optimizer_type == "adam":
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

		train_length = int(train_percentage * len(dataset))
		test_length = len(dataset) - train_length
		train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length))
		self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.AUTOENCODER.BATCH_SIZE, shuffle=True)
		self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.AUTOENCODER.BATCH_SIZE, shuffle=True)
		self.current_iteration = 0
		self.current_epoch = 0
		self.vae_feature = self.dataset.vae_feature

	def train(self):
		self.model.train()
		self.model.to(self.device)
		running_total_loss = []
		running_reconstruction_loss = []
		running_kld_loss = []
		evaluation_files = glob.glob("datasets/vae/evaluation/{}/*.png".format(self.vae_feature))
		for filePath in evaluation_files:
			try:
				os.remove(filePath)
			except:
				print("Error while deleting file : ", filePath)

		for epoch in range(1, self.epochs + 1):
			for i, data in enumerate(self.train_dl):
				self.current_iteration = i
				data = data.to(self.device)
				self.optimizer.zero_grad()
				reconstruction, mu, log_var = self.model(data)
				mu = torch.where(mu.double() > self.cfg.AUTOENCODER.MAX_MEAN, self.cfg.AUTOENCODER.MAX_MEAN,
								 mu.double())
				log_var = torch.where(log_var.double() > self.cfg.AUTOENCODER.MAX_VAR, self.cfg.AUTOENCODER.MAX_VAR,
								 log_var.double())
				loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var)
				running_total_loss.append(loss.item())
				running_reconstruction_loss.append(recon_loss.item())
				running_kld_loss.append(kld_loss.item())
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.AUTOENCODER.MAX_GRADIENT)
				self.optimizer.step()
				if not i % self.cfg.AUTOENCODER.LOG_INTERVAL and i > 0:
					self.save_images(data, reconstruction, i, epoch, "train")
					print("[{}/{}] Train reconstruction loss: {}".format(i, int(
						len(self.train_dl.dataset) / self.train_dl.batch_size), (sum(running_reconstruction_loss) / len(
						running_reconstruction_loss))))
					print("[{}/{}] Train kld loss: {}".format(i, int(
						len(self.train_dl.dataset) / self.train_dl.batch_size),
															  (sum(running_kld_loss) / len(running_kld_loss))))
					print("[{}/{}] Train total loss: {} \n".format(i, int(
						len(self.train_dl.dataset) / self.train_dl.batch_size),
																   (sum(running_total_loss) / len(running_total_loss))))

			train_total_loss = sum(running_total_loss) / len(running_total_loss)
			train_reconstruction_loss = sum(running_reconstruction_loss) / len(running_reconstruction_loss)
			train_kld_loss = sum(running_kld_loss) / len(running_kld_loss)
			self.current_epoch = epoch
			print("Epoch {}: Train reconstruction loss: {}".format(self.current_epoch, train_reconstruction_loss))
			print("Epoch {}: Train kld loss: {}".format(self.current_epoch, train_kld_loss))
			print("Epoch {}: Train total loss: {} \n".format(self.current_epoch, train_total_loss))

			plt.clf()
			plt.axis("on")
			# plt.legend(["total loss", "reconstruction loss", "kld loss"])
			# plt.plot(running_total_loss)
			plt.plot(running_reconstruction_loss)
			# plt.plot(running_kld_loss)
			plt.ylabel("train reconstruction_loss")
			plt.xlabel("# iterations")
			# plt.yscale("log")
			plt.ylim(0, 1)
			plt.title("Train Loss in Epoch {}/{} ".format(self.current_epoch, self.epochs))
			plt.savefig(
				"datasets/vae/evaluation/{}/train_loss_curve_{}.png".format(self.vae_feature, self.current_epoch))

			test_loss = self.test()

			torch.save(self.model.state_dict(),
					   "datasets/vae/vae_{}_model.pt".format(self.vae_feature))

		return train_total_loss, test_loss

	def loss(self, reconstruction, input, mu, log_var):
		if self.loss_function == "l1":
			recons_loss = torch.nn.functional.l1_loss(reconstruction, input, reduction="mean")
		else:
			recons_loss = torch.nn.functional.mse_loss(reconstruction, input)

		kld_weight = 1
		if self.current_epoch == 0:
			kld_weight = self.current_iteration / len(self.train_dl)

		kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
		kld_loss /= self.cfg.AUTOENCODER.LATENT_SPACE
		kld_loss *= kld_weight
		loss = recons_loss + kld_loss
		return loss, recons_loss, kld_loss

	def test(self):
		self.model.eval()
		self.model.to(self.device)
		running_total_loss = []
		running_reconstruction_loss = []
		running_kld_loss = []
		with torch.no_grad():
			for i, data in enumerate(self.test_dl):
				data = data.to(self.device)
				reconstruction, mu, log_var = self.model(data)
				loss, recon_loss, kld_loss = self.loss(reconstruction, data, mu, log_var)
				running_total_loss.append(loss.item())
				running_reconstruction_loss.append(recon_loss.item())
				running_kld_loss.append(kld_loss.item())
				if not i % self.cfg.AUTOENCODER.LOG_INTERVAL and i > 0:
					self.save_images(data, reconstruction, i, self.current_epoch, "test")

			test_total_loss = sum(running_total_loss) / len(running_total_loss)
			test_reconstruction_loss = sum(running_reconstruction_loss) / len(running_reconstruction_loss)
			test_kld_loss = sum(running_kld_loss) / len(running_kld_loss)
			print("Epoch {}: Test reconstruction loss: {}".format(self.current_epoch, test_reconstruction_loss))
			print("Epoch {}: Test kld loss: {}".format(self.current_epoch, test_kld_loss))
			print("Epoch {}: Test total loss: {} \n".format(self.current_epoch, test_total_loss))

			plt.clf()
			plt.axis("on")
			# plt.legend(["total loss", "reconstruction loss", "kld loss"])
			# plt.plot(running_total_loss)
			plt.plot(running_reconstruction_loss)
			# plt.plot(running_kld_loss)
			plt.ylabel("test reconstruction_loss")
			plt.xlabel("# iterations")
			# plt.yscale("log")
			plt.ylim(0, 1)
			plt.title("Test Loss over in Epoch {}/{} ".format(self.current_epoch, self.epochs))
			plt.savefig(
				"datasets/vae/evaluation/{}/test_loss_curve_{}.png".format(self.vae_feature, self.current_epoch))
			return test_total_loss

	def save_images(self, inputs, reconstructions, iteration, epoch, prefix):
		for i in range(len(inputs)):
			original_image = None
			reconstruction_image = None
			original_item = inputs[i][0]
			reconstruction_item = reconstructions[i][0]
			for j in range(0, len(original_item), 10):
				if original_image is None:
					original_image = original_item[j].detach().cpu().numpy()
					reconstruction_image = reconstruction_item[j].detach().cpu().numpy()
				else:
					original_image = np.concatenate((original_image, original_item[j].detach().cpu().numpy()), 0)
					reconstruction_image = np.concatenate(
						(reconstruction_image, reconstruction_item[j].detach().cpu().numpy()), 0)

			# reconstruction_image -= reconstruction_image.min()
			# reconstruction_image /= reconstruction_image.max()
			evaluation_image = np.concatenate((original_image, reconstruction_image), 1)

			plt.axis('off')
			plt.imsave(
				'datasets/vae/evaluation/' + self.vae_feature + '/{}_{}_{}.png'.format(prefix, epoch, iteration + i),
				evaluation_image,
				cmap="gray")
			return

	def save_latent_feature(self):
		self.model.load_state_dict(
			torch.load("datasets/vae/" + "vae_{}_model.pt".format(self.vae_feature)))
		self.model.eval()
		self.model.to(self.device)
		all_regions = self.dataset.prep_data_info(save=False)

		with h5py.File("datasets/vae/" + "vae_data_{}.h5".format(self.cfg.AUTOENCODER.TARGET[0]), 'r') as volume_file:
			with h5py.File("features/{}f.h5".format(self.vae_feature), 'w') as f:
				if self.vae_feature in f.keys():
					del f[self.vae_feature]
				if "id" in f.keys():
					del f["id"]
				f.create_dataset(name=self.vae_feature,
								 shape=(len(all_regions), self.cfg.AUTOENCODER.LATENT_SPACE))
				f.create_dataset(name="id",
								 shape=(len(all_regions),))
				with torch.no_grad():
					for i, region in tqdm(enumerate(all_regions), total=len(all_regions)):
						x = torch.zeros(self.cfg.AUTOENCODER.LATENT_SPACE)
						for j, vid in enumerate(volume_file["id"]):
							if vid == region["id"]:
								volume = volume_file[self.vae_feature + "_volume"][j]
								volume = np.expand_dims(volume, axis=0)
								data = torch.from_numpy(volume).cuda()
								data.to(self.device)
								x = self.model.latent_representation(data).cpu().numpy()

						f[self.vae_feature][i] = x
						f["id"][i] = region["id"]


######################################################
#### ---- Pointcloud based Learning section ----- ####
######################################################
class PtcTrainer():
	'''
	Training object for the pointcloud Autoencoder.
	:params cfg: configuration sheet.
	:params dataloader: related dataloader class object.
	:params train_percentage: (float) split dataset into train and test.
	:params optimizer_type: optimization algorithm (default: Adam)
	'''
	def __init__(self, cfg, dataset, train_percentage=0.7, optimizer_type='adam'):
		self.cfg = cfg
		self.dataset = dataset
		self.train_percentage = train_percentage
		self.optimizer_type = optimizer_type
		self.dist = ChamferDistance()
		self.num_points = self.cfg.PTC.RECON_NUM_POINTS
		#self.model_type = cfg.PTC.ARCHITECTURE
		self.vae_ptc_feature = self.cfg.PTC.FEATURE_NAME
		self.epochs = self.cfg.PTC.EPOCHS
		self.device = self.cfg.PTC.DEVICE

		# Setting the outputpath for each run.
		time_now = str(datetime.datetime.now()).split(' ')
		if os.path.exists(os.path.join(self.cfg.PTC.MONITOR_PATH, 'run_' + time_now[0])):
			self.output_path = os.path.join(self.cfg.PTC.MONITOR_PATH, 'run_' + time_now[0])
		else:
			self.output_path = os.path.join(self.cfg.PTC.MONITOR_PATH, 'run_' + time_now[0])
			os.mkdir(self.output_path)

		train_length = int(train_percentage * len(self.dataset))
		train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, (train_length, len(self.dataset) - train_length))
		self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.PTC.BATCH_SIZE, shuffle=False)
		self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.PTC.BATCH_SIZE, shuffle=False)

		self.model = PTCvae(num_points=self.num_points, latent_space=self.cfg.PTC.LATENT_SPACE)
		if self.optimizer_type == "adam":
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.PTC.LR, weight_decay=self.cfg.PTC.WEIGHT_DECAY)

	def train(self):
		self.model.train()
		self.model.to(self.device)

		counter = 0
		self.logger = build_monitor(self.cfg, self.output_path, 'train')
		running_loss = list()
		for epoch in range(1, self.epochs + 1):
			for i, data in enumerate(self.train_dl):
				data, y = data
				data = data.to(self.device).float()
				self.optimizer.zero_grad()
				x = self.model(data)
				loss = self.loss(x, data)

				running_loss.append(loss.item())
				loss.backward()
				self.optimizer.step()

				if (not i % self.cfg.PTC.LOG_INTERVAL or i == 1) and i > 0:
					print("[{}/{}] Train total loss: {} \n".format(i, int(\
					len(self.train_dl.dataset) / self.train_dl.batch_size),\
					(sum(running_loss) / len(running_loss))))
					self.logger.update((sum(running_loss) / len(running_loss)), counter, self.cfg.PTC.LR, epoch)
				counter = counter + 1

			self.current_epoch = epoch
			train_total_loss = sum(running_loss) / len(running_loss)
			print("Epoch {}: Train total loss: {} \n".format(self.current_epoch, train_total_loss))

			test_loss = self.test()
			torch.save(self.model.state_dict(), os.path.join(self.output_path, 'vae_ptc_model_{}.pt'.format(epoch)))

		print('Training and Testing of the point cloud based autoencoder is done.')
		print("train loss: {}".format(train_total_loss))
		print("test loss: {}".format(test_loss))

	def test(self):
		self.model.eval()
		self.model.to(self.device)
		running_loss = list()
		counter = 0
		self.logger = build_monitor(self.cfg,self.output_path, 'test')
		with torch.no_grad():
			for i, data in enumerate(self.test_dl):
				data, y = data
				data = data.to(self.device).float()
				x = self.model(data)
				loss = self.loss(x, data)
				running_loss.append(loss.item())

				if not i % self.cfg.PTC.LOG_INTERVAL and i > 0:
					self.logger.update((sum(running_loss) / len(running_loss)), counter, self.cfg.PTC.LR)
				counter = counter + 1

			test_total_loss = sum(running_loss) / len(running_loss)
			print("Epoch {}: Test total loss: {} \n".format(self.current_epoch, test_total_loss))

			return test_total_loss

	def loss(self, reconstruction, org_data):
		rec = torch.squeeze(reconstruction, axis=0)
		org = torch.squeeze(org_data, axis=0)
		rec_loss = self.dist(rec, org)
		return rec_loss

	def save_latent_feature(self, m_version: int = 5):
		'''saving the latent space representation of every point cloud.'''
		self.model.load_state_dict(torch.load(os.path.join(self.output_path, 'vae_ptc_model_{}.pt'.format(m_version))))
		self.model.eval()
		self.model.to(self.device)

		whole_ds = torch.utils.data.DataLoader(self.dataset)
		with h5py.File('features/{}.h5'.format(self.vae_ptc_feature), 'w') as h5f:
			h5f.create_dataset(name='ptc_shape', shape=(len(self.dataset.keys), self.cfg.PTC.LATENT_SPACE))
			h5f.create_dataset(name='id', shape=(len(self.dataset.keys),))

			with torch.no_grad():
				for i, data in tqdm(enumerate(whole_ds), total=len(self.dataset.keys)):
					data, y = data
					data = data.to(self.device).float()

					x = self.model.latent_representation(data)
					recon = self.model.latent_recon(x)
					self.save_ptcs(recon, y[0])
					x = x.cpu().numpy()

					h5f['ptc_shape'][i] = x
					h5f['id'][i] = int(y[0])
			h5f.close()

	def save_ptcs(self, reconstructions, idx, save=True):
		'''Save the reconstructed point clouds to h5.'''
		rec_ptc = reconstructions.view(reconstructions.size(2), reconstructions.size(3))
		ptc = rec_ptc.detach().numpy()
		if os.path.exists(self.cfg.PTC.MONITOR_PATH + self.cfg.PTC.RECONSTRUCTION_DATA) is False:
			with h5py.File(self.cfg.PTC.MONITOR_PATH + self.cfg.PTC.RECONSTRUCTION_DATA, 'w') as h5f:
				grp = h5f.create_group('rec_ptc')
				grp.create_dataset(idx, data=ptc)
				h5f.close()
		else:
			with h5py.File(self.cfg.PTC.MONITOR_PATH + self.cfg.PTC.RECONSTRUCTION_DATA, 'r+') as h5f:
				grp = h5f.get(list(h5f.keys())[0])
				grp.create_dataset(idx, data=ptc)
				h5f.close()

	def visualize_single_ptc(self, x):
		'''visualize one single point cloud in training process.'''
		tmp = torch.squeeze(x).detach().numpy()
		visptc(tmp)

	# def visualise_ptcs(self, model):
	# 	model.eval()
	# 	model.to(self.device)
	#
	# 	with torch.no_grad():
	# 		for c, idx in enumerate(self.dataset.keys):
	# 			data = torch.from_numpy(self.dataset[idx])
	# 			data = data.unsqueeze(0).float()
	# 			data.to(self.device)
	# 			x = model.latent_representation(data).cpu().numpy()
	# 			print(x.shape)
	# 			if idx == 1:
	# 				break

def random_ptc_infer(model, dataset):
	ptc_datamodule = RandomPtcDataModule(cfg=dataset.cfg, dataset=dataset)
	ptc_datamodule.setup()
	ptc_dataloader = ptc_datamodule.train_dataloader()
	keys = dataset.keys
	with h5py.File('features/{}.h5'.format(self.vae_ptc_feature), 'w') as f:
		f.create_dataset(name='id', shape=(len(keys),))
		for i, x in tqdm(enumerate(ptc_dataloader), total=len(keys)):
			f['id'][i] = i
			latent_space = model.save_latent(x)
			if 'ptc_shape' not in f.keys():
				f.create_dataset(name='ptc_shape', shape=(len(keys), latent_space.shape[-1]))
			f['ptc_shape'][i] = latent_space
		f.close()
