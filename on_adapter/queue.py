"""Safe entrypoint: a plan is the default; execution requires explicit authorization."""
def main():
    from .scheduler import main as dispatch
    dispatch()
if __name__ == '__main__': main()
