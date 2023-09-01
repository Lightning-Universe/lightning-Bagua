Search.setIndex({"docnames": ["index", "overview"], "filenames": ["index.rst", "overview.rst"], "titles": ["Lightning \u26a1 Bagua", "Lightning \u26a1 Bagua"], "terms": {"i": [0, 1], "deep": [0, 1], "learn": [0, 1], "train": [0, 1], "acceler": [0, 1], "framework": [0, 1], "which": [0, 1], "support": [0, 1], "multipl": [0, 1], "advanc": [0, 1], "distribut": [0, 1], "algorithm": [0, 1], "includ": [0, 1], "gradient": [0, 1], "allreduc": [0, 1], "central": [0, 1], "synchron": [0, 1], "commun": [0, 1], "where": [0, 1], "ar": [0, 1], "averag": [0, 1], "among": [0, 1], "all": [0, 1], "worker": [0, 1], "decentr": [0, 1], "sgd": [0, 1], "each": [0, 1], "exchang": [0, 1], "data": [0, 1], "one": [0, 1], "few": [0, 1], "specif": [0, 1], "bytegrad": [0, 1], "qadam": [0, 1], "low": [0, 1], "precis": [0, 1], "compress": [0, 1], "befor": [0, 1], "asynchron": [0, 1], "model": [0, 1], "requir": [0, 1], "same": [0, 1], "iter": [0, 1], "lock": [0, 1], "step": [0, 1], "style": [0, 1], "By": [0, 1], "default": [0, 1], "us": [0, 1], "also": [0, 1], "implement": [0, 1], "ddp": [0, 1], "can": [0, 1], "usual": [0, 1], "produc": [0, 1], "higher": [0, 1], "throughput": [0, 1], "due": [0, 1], "its": [0, 1], "backend": [0, 1], "written": [0, 1], "rust": [0, 1], "instal": [0, 1], "pip": [0, 1], "u": [0, 1], "usag": [0, 1], "simpli": [0, 1], "set": [0, 1], "strategi": [0, 1], "argument": [0, 1], "trainer": [0, 1], "from": [0, 1], "import": [0, 1], "4": [0, 1], "gpu": [0, 1], "mode": [0, 1], "devic": [0, 1], "specifi": [0, 1], "baguastrategi": [0, 1], "you": [0, 1], "select": [0, 1], "more": [0, 1], "featur": [0, 1], "lightning_bagua": [0, 1], "gradient_allreduc": [0, 1], "low_precision_decentr": [0, 1], "interv": [0, 1], "100m": [0, 1], "async": [0, 1], "sync_interval_m": [0, 1], "100": [0, 1], "To": [0, 1], "we": [0, 1], "need": [0, 1], "initi": [0, 1], "qadamoptim": [0, 1], "first": [0, 1], "l": [0, 1], "torch_api": [0, 1], "q_adam": [0, 1], "class": [0, 1], "mymodel": [0, 1], "lightningmodul": [0, 1], "def": [0, 1], "configure_optim": [0, 1], "self": [0, 1], "optim": [0, 1], "return": [0, 1], "paramet": [0, 1], "lr": [0, 1], "0": [0, 1], "05": [0, 1], "warmup_step": [0, 1], "fit": [0, 1], "reli": [0, 1], "own": [0, 1], "launcher": [0, 1], "schedul": [0, 1], "job": [0, 1], "below": [0, 1], "find": [0, 1], "exampl": [0, 1], "launch": [0, 1], "follow": [0, 1], "torch": [0, 1], "api": [0, 1], "start": [0, 1], "8": [0, 1], "singl": [0, 1], "node": [0, 1], "python": [0, 1], "m": [0, 1], "nproc_per_nod": [0, 1], "py": [0, 1], "If": [0, 1], "ssh": [0, 1], "servic": [0, 1], "avail": [0, 1], "passwordless": [0, 1], "login": [0, 1], "baguarun": [0, 1], "ha": [0, 1], "similar": [0, 1], "syntax": [0, 1], "mpirun": [0, 1], "when": [0, 1], "stare": [0, 1], "automat": [0, 1], "spawn": [0, 1], "new": [0, 1], "process": [0, 1], "your": [0, 1], "provid": [0, 1], "host_list": [0, 1], "option": [0, 1], "describ": [0, 1], "an": [0, 1], "ip": [0, 1], "address": [0, 1], "port": [0, 1], "run": [0, 1], "node1": [0, 1], "node2": [0, 1], "two": [0, 1], "per": [0, 1], "hostname1": [0, 1], "ssh_port1": [0, 1], "hostname2": [0, 1], "ssh_port2": [0, 1], "master_port": [0, 1], "port1": [0, 1], "wai": [0, 1], "parallel": [0, 1], "howev": [0, 1], "system": [0, 1], "like": [0, 1], "net": [0, 1], "http": [0, 1], "tutori": [0, 1], "baguasi": [0, 1], "com": [0, 1], "perform": [0, 1], "autotun": [0, 1], "onli": [0, 1], "enabl": [0, 1], "through": [0, 1], "It": [0, 1], "worth": [0, 1], "note": [0, 1], "achiev": [0, 1], "better": [0, 1], "without": [0, 1], "modifi": [0, 1], "script": [0, 1], "see": [0, 1], "detail": [0, 1]}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"lightn": [0, 1], "bagua": [0, 1]}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Lightning \u26a1 Bagua": [[0, "lightning-bagua"], [1, "lightning-bagua"]]}, "indexentries": {}})