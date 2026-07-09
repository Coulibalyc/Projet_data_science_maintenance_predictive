"""
Lance les dashboards HTML/CSS/JS.
    python3 launch.py [--port 8765] [--no-browser] [--skip-export]
"""
import argparse, os, subprocess, sys, webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

parser = argparse.ArgumentParser()
parser.add_argument('--port',        type=int,  default=8765)
parser.add_argument('--no-browser',  action='store_true')
parser.add_argument('--skip-export', action='store_true', help='Ne pas regénérer les dashboards')
args = parser.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# 1. Génération des dashboards
if not args.skip_export:
    print("Génération des dashboards…")
    python = os.path.join(ROOT, 'venv', 'bin', 'python') if os.path.exists(
        os.path.join(ROOT, 'venv', 'bin', 'python')) else sys.executable
    result = subprocess.run([python, 'generate_dashboards.py'], capture_output=False)
    if result.returncode != 0:
        print("[ERREUR] generate_dashboards.py a échoué. Vérifiez vos dépendances.")
        sys.exit(1)
else:
    print("[skip-export] Utilisation des HTML existants.")

# 2. Serveur HTTP
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silencieux sauf erreurs

print(f"\n{'='*50}")
print(f"  Dashboard Client    → http://localhost:{args.port}/dashboard_client.html")
print(f"  Dashboard Technique → http://localhost:{args.port}/dashboard_technique.html")
print(f"{'='*50}")
print("  Ctrl+C pour arrêter\n")

if not args.no_browser:
    webbrowser.open(f"http://localhost:{args.port}/dashboard_client.html")

try:
    httpd = HTTPServer(('localhost', args.port), QuietHandler)
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n  Serveur arrêté.")
