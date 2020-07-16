import os

import torch
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model as m
from utils.plotting import plot_ticker
from utils.data_wrapper import StockData
from config import get_config, print_usage, print_config


def get_model(config):
    if config.model_type == "LSTM":
        return m.LSTM
    elif config.model_type == "GRU":
        return m.GRU
    elif config.model_type == "LSTMForcing":
        return m.GRU


def model_criterion(config):
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param ** 2)

        return loss * config.l2_reg

    return model_loss


def train(config):
    # TODO: Change input to be an optimizable torch variable?
    # TODO: Train on price *differences* versus absolute price values (even if they are scaled to -1,1)
    # TODO: Include volume?
    # TODO: Add model dropout

    # does this ever change? I don't think so?
    input_dim = 1

    config.train = "./data/mini"
    config.val = "./data/mini"

    train_data = StockData(config.train, config.rolling_mean, config.input_length, config.output_length)

    # the scaler for the input data
    scaler = train_data.get_scaler()

    val_data = StockData(config.val, config.rolling_mean, config.input_length, config.output_length,
                         scaler=scaler)

    print(f"Training shape:{train_data.data.shape}")
    print(f"Validation shape:{val_data.data.shape}")

    tr_data_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=val_data,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False
    )

    model = get_model(config)(input_dim=input_dim,
                              hidden_dim=config.hidden_dim,
                              output_dim=config.output_length,
                              num_layers=config.num_layers,
                              dropout=config.dropout)

    model_loss = model_criterion(config)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()

    tr_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "train"))
    va_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "valid"))

    iter_idx = -1
    best_va_loss = np.inf
    best_model_file = os.path.join(config.save_dir, f"best_{config.model_type}.pth")

    model.train()
    hist = np.zeros(config.epochs)

    for epoch in range(config.epochs):
        prefix = "Training Epoch {:3d}: ".format(epoch)
        for data in tqdm(tr_data_loader, desc=prefix, postfix="loss {:8f}".format(best_va_loss)):
            iter_idx += 1
            # Split the data
            x, y, ticker = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            predictions = model(x)

            data_loss = criterion(predictions, y.type(torch.float32))
            m_loss = model_loss(model)
            loss = data_loss + m_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # x = x.squeeze().cpu()
            # y = y.squeeze().cpu()
            # predictions = predictions.cpu()
            # plot_ticker(x, y, predictions, ticker=ticker)

            hist[epoch] += loss.item()

            if iter_idx % config.rep_intv == 0:
                tr_writer.add_scalar("loss", loss, global_step=iter_idx)
                tr_writer.flush()

        if epoch % config.val_intv == 0:
            va_loss = []
            model = model.eval()
            for data_val in val_data_loader:
                x, y, ticker = data_val

                # Send data to GPU if we have one
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                with torch.no_grad():
                    predictions = model(x)
                    loss = criterion(predictions, y.type(torch.float32)) + model_loss(model)
                    va_loss += [loss.cpu().numpy()]

            model = model.train()
            va_loss = np.mean(va_loss)
            va_writer.add_scalar("loss", va_loss, global_step=iter_idx)
            va_writer.flush()

            # save the best model at the lowest loss, there isn't a difference accuracy metric for a time series
            if va_loss < best_va_loss:
                print("Saved")
                best_va_loss = va_loss
                torch.save({
                    "iter_idx": iter_idx,
                    "best_va_loss": best_va_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler,
                }, best_model_file)


    import matplotlib.pyplot as plt
    plt.plot(hist)
    plt.show()


def test(config):
    input_dim = 1

    best_model_file = os.path.join(config.save_dir, f"best_{config.model_type}.pth")
    load_res = torch.load(best_model_file)
    scaler = load_res["scaler"]

    config.test = "./data/mini_test"
    test_data = StockData(config.test, config.rolling_mean, config.input_length, config.output_length, scaler=scaler,
                          single_file=config.test_file)

    print(f"Testing shape:{test_data.data.shape}")

    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False
    )

    model = get_model(config)(input_dim=input_dim,
                              hidden_dim=config.hidden_dim,
                              output_dim=config.output_length,
                              num_layers=config.num_layers,
                              dropout=config.dropout)

    model_loss = model_criterion(config)
    criterion = torch.nn.MSELoss(reduction='mean')

    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(load_res["model"])
    model.eval()

    test_loss = []

    prefix = "Training: "
    counter = 0
    for data in tqdm(test_data_loader, desc=prefix):
        x, y, ticker = data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            predictions = model(x)
            loss = criterion(predictions, y.type(torch.float32)) + model_loss(model)
            test_loss += [loss.cpu().numpy()]

            x = x.squeeze().cpu()
            y = y.squeeze().cpu()
            predictions = predictions.cpu()
            plot_ticker(x, y, predictions, scaler=scaler, ticker=ticker, batch=counter)
            counter += 1

    print("Test Loss = {}".format(np.mean(test_loss)))


def main(config):
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)


if __name__ == "__main__":
    cfg, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    print_config(cfg)
    main(cfg)
