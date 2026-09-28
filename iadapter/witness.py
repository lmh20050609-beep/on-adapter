import torch
if __name__ == '__main__':
    torch.manual_seed(0)
    x=torch.randn(8,8,device='cuda')
    y=x@x
    assert torch.isfinite(y).all()
    print('WITNESS',tuple(y.shape),torch.cuda.get_device_name())
