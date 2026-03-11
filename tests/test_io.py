import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from data.dataset import create_dataloaders
import argparse

def test_cache():
    # Use relative testing data path
    root = PROJECT_ROOT / "data" / "MAP1"
    if not root.exists():
        print(f"Skipping test: {root} does not exist")
        return

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=[root],
        batch_size=4,
        num_workers=0,
        val_split=0.2,
        split_mode="tile" # Just sample from one map
    )
    
    print(f"Train loader has {len(train_loader)} batches")
    print("Testing train batch...")
    for idx, batch in enumerate(train_loader):
        print(f"Batch {idx} loaded: img shape {batch['image'].shape}")
        if idx >= 1: # Just test a couple
            break

if __name__ == "__main__":
    main()
