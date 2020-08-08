        def minmax(net):
                a = 1e50
                b = -1e50
                for p in net.parameters():
                    a = min(a, p.data.min().item())
                    b = max(b, p.data.max().item())
                    if (p.data != p.data).any():
                        return "NAN", "NAN"
                return a,b