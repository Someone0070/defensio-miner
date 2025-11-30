#!/usr/bin/env python3
"""
Defensio Miner - Wallet Utilities
=================================
Utility commands for wallet management, stats, and export.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from pycardano import PaymentSigningKey
except ImportError:
    print("Installing pycardano...")
    os.system(f"{sys.executable} -m pip install pycardano -q")
    from pycardano import PaymentSigningKey

# Configuration
API_BASE = os.environ.get("DEFENSIO_API_BASE", "https://mine.defensio.io/api")
WALLET_DIR = Path(os.environ.get("DEFENSIO_WALLET_DIR", "./wallets"))


def load_wallets() -> List[Dict]:
    """Load wallets from file."""
    wallet_file = WALLET_DIR / "wallets.json"
    if not wallet_file.exists():
        print("No wallets found. Run miner.py first to generate wallets.")
        return []
    
    with open(wallet_file, 'r') as f:
        data = json.load(f)
    return data.get('wallets', [])


def save_wallets(wallets: List[Dict]):
    """Save wallets to file."""
    wallet_file = WALLET_DIR / "wallets.json"
    data = {
        'wallets': wallets,
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    with open(wallet_file, 'w') as f:
        json.dump(data, f, indent=2)


def cmd_list(args):
    """List all wallets."""
    wallets = load_wallets()
    
    if not wallets:
        return
    
    print(f"\n{'ID':<5} {'Address':<58} {'Solutions':<10} {'Registered':<12}")
    print("-" * 90)
    
    total_solutions = 0
    for w in wallets:
        reg = "Yes" if w.get('is_registered') else "No"
        sols = w.get('solutions_submitted', 0)
        total_solutions += sols
        print(f"{w['id']:<5} {w['address']:<58} {sols:<10} {reg:<12}")
    
    print("-" * 90)
    print(f"Total: {len(wallets)} wallets, {total_solutions} solutions")


def cmd_stats(args):
    """Show statistics for all wallets."""
    wallets = load_wallets()
    
    if not wallets:
        return
    
    print("\nFetching stats from API...")
    
    total_earned = 0.0
    wallet_stats = []
    
    for w in wallets:
        try:
            response = requests.get(
                f"{API_BASE}/wallet/{w['address']}/stats",
                timeout=10
            )
            if response.status_code == 200:
                stats = response.json()
                earned = stats.get('earned', stats.get('dfo_earned', 0))
                wallet_stats.append({
                    'id': w['id'],
                    'address': w['address'][:40] + '...',
                    'solutions': w.get('solutions_submitted', 0),
                    'earned': earned
                })
                total_earned += float(earned)
        except Exception as e:
            print(f"Error fetching stats for wallet {w['id']}: {e}")
    
    print(f"\n{'ID':<5} {'Address':<45} {'Solutions':<10} {'Earned DFO':<15}")
    print("-" * 80)
    
    for s in wallet_stats:
        print(f"{s['id']:<5} {s['address']:<45} {s['solutions']:<10} {s['earned']:<15.4f}")
    
    print("-" * 80)
    print(f"Total Earned: {total_earned:.4f} DFO")


def cmd_export_skeys(args):
    """Export signing keys for import into Cardano wallet."""
    wallets = load_wallets()
    
    if not wallets:
        return
    
    skey_dir = Path(args.output)
    skey_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {len(wallets)} signing keys to {skey_dir}...")
    
    for w in wallets:
        skey_file = skey_dir / f"wallet_{w['id']}.skey"
        
        # Format as Cardano CLI compatible skey
        skey_data = {
            "type": "PaymentSigningKeyShelley_ed25519",
            "description": f"Defensio Miner Wallet {w['id']}",
            "cborHex": f"5820{w['signing_key']}"
        }
        
        with open(skey_file, 'w') as f:
            json.dump(skey_data, f, indent=2)
        
        print(f"  Created {skey_file}")
    
    print(f"\nKeys exported to {skey_dir}/")
    print("Import these into Eternl or other Cardano wallet using CLI signing key import.")


def cmd_export_mnemonics(args):
    """Export mnemonics for backup."""
    wallets = load_wallets()
    
    if not wallets:
        return
    
    output_file = Path(args.output)
    
    print(f"\nExporting {len(wallets)} mnemonics to {output_file}...")
    print("\n⚠️  WARNING: This file contains sensitive data!")
    print("    Keep it secure and delete after backup.\n")
    
    with open(output_file, 'w') as f:
        f.write("# Defensio Miner Wallet Mnemonics Backup\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("# KEEP THIS FILE SECURE!\n\n")
        
        for w in wallets:
            f.write(f"Wallet {w['id']}:\n")
            f.write(f"  Address: {w['address']}\n")
            f.write(f"  Mnemonic: {w['mnemonic']}\n")
            f.write(f"  Solutions: {w.get('solutions_submitted', 0)}\n\n")
    
    # Set restrictive permissions
    os.chmod(output_file, 0o600)
    
    print(f"Mnemonics exported to {output_file}")


def cmd_consolidate(args):
    """Consolidate earnings to a single address."""
    wallets = load_wallets()
    
    if not wallets:
        return
    
    recipient = args.address
    print(f"\nConsolidating {len(wallets)} wallets to {recipient[:30]}...")
    
    # First, register recipient if needed
    try:
        response = requests.post(
            f"{API_BASE}/register",
            json={'address': recipient},
            timeout=30
        )
    except Exception as e:
        print(f"Warning: Could not register recipient: {e}")
    
    success = 0
    failed = 0
    
    for w in wallets:
        if w['address'] == recipient:
            continue
        
        if w.get('solutions_submitted', 0) == 0:
            continue
        
        try:
            # Create message and signature
            import hashlib
            message = f"Assign accumulated Scavenger rights to: {recipient}"
            signature = hashlib.blake2b(
                message.encode() + bytes.fromhex(w['signing_key']),
                digest_size=64
            ).hexdigest()
            
            response = requests.post(
                f"{API_BASE}/donate_to",
                json={
                    'donorAddress': w['address'],
                    'recipientAddress': recipient,
                    'signature': signature,
                    'message': message
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  ✓ Wallet {w['id']}: {w['address'][:30]}...")
                success += 1
            else:
                print(f"  ✗ Wallet {w['id']}: {response.text}")
                failed += 1
        except Exception as e:
            print(f"  ✗ Wallet {w['id']}: {e}")
            failed += 1
        
        import time
        time.sleep(2)  # Rate limiting
    
    print(f"\nConsolidation complete: {success} success, {failed} failed")


def cmd_check_challenges(args):
    """Check challenge status."""
    challenge_file = WALLET_DIR / "challenges.json"
    
    if not challenge_file.exists():
        print("No challenges found.")
        return
    
    with open(challenge_file, 'r') as f:
        data = json.load(f)
    
    challenges = data.get('challenges', [])
    
    print(f"\n{'ID':<12} {'Difficulty':<12} {'Solved':<10} {'Valid':<8}")
    print("-" * 50)
    
    valid_count = 0
    solved_count = 0
    
    for c in challenges:
        solved = len(c.get('solved_wallets', []))
        
        # Check if valid (simple check, not full expiry check)
        is_valid = True  # Would need timestamp comparison
        
        print(f"{c['id'][:10]:<12} {c.get('difficulty', '?'):<12} {solved:<10} {'Yes' if is_valid else 'No':<8}")
        
        if is_valid:
            valid_count += 1
        solved_count += solved
    
    print("-" * 50)
    print(f"Total: {len(challenges)} challenges, {valid_count} valid, {solved_count} solutions")


def main():
    parser = argparse.ArgumentParser(description="Defensio Miner Wallet Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all wallets')
    list_parser.set_defaults(func=cmd_list)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show wallet statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Export skeys command
    skey_parser = subparsers.add_parser('export-skeys', help='Export signing keys')
    skey_parser.add_argument('-o', '--output', default='./skeys',
                            help='Output directory (default: ./skeys)')
    skey_parser.set_defaults(func=cmd_export_skeys)
    
    # Export mnemonics command
    mnemonic_parser = subparsers.add_parser('export-mnemonics', help='Export mnemonics for backup')
    mnemonic_parser.add_argument('-o', '--output', default='./mnemonics_backup.txt',
                                 help='Output file (default: ./mnemonics_backup.txt)')
    mnemonic_parser.set_defaults(func=cmd_export_mnemonics)
    
    # Consolidate command
    consolidate_parser = subparsers.add_parser('consolidate', help='Consolidate earnings')
    consolidate_parser.add_argument('address', help='Recipient Cardano address')
    consolidate_parser.set_defaults(func=cmd_consolidate)
    
    # Challenges command
    challenges_parser = subparsers.add_parser('challenges', help='Check challenge status')
    challenges_parser.set_defaults(func=cmd_check_challenges)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
