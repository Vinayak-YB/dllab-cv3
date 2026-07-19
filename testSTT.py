with torch.no_grad():

    input_seq, target = next(iter(loader))

    input_seq = input_seq.to(device)
    target = target.squeeze(1).to(device)

    z4 = ae.encode(input_seq[:, -1])
    z5 = ae.encode(target)

    print(torch.mean((z4 - z5) ** 2))