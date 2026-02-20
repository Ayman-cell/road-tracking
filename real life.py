"""
╔══════════════════════════════════════════════════════════════════╗
║          🚗 VEHICLE COUNTER — YOLO26s Dashboard Python          ║
║          Interface graphique complète (Tkinter + OpenCV)        ║
╚══════════════════════════════════════════════════════════════════╝

INSTALLATION (copie-colle dans ton terminal) :
    pip install ultralytics opencv-python pillow

LANCER :
    python vehicle_counter_gui.py
"""

import tkinter as tk
from tkinter import ttk, font
import cv2
import time
import threading
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
SOURCE        = 0        # 0 = webcam | "video.mp4" = fichier
CONF_THRESH   = 0.35
IMGSZ         = 416
LINE_RATIO    = 0.55

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
ICONS  = {"Car": "🚗", "Motorcycle": "🏍", "Bus": "🚌", "Truck": "🚛"}

# Palette de couleurs (hex pour Tkinter, BGR pour OpenCV)
COLORS_HEX = {
    "Car":        "#00d4ff",
    "Motorcycle": "#00ff88",
    "Bus":        "#ff9500",
    "Truck":      "#bf5fff",
}
COLORS_BGR = {
    "Car":        (255, 212, 0),
    "Motorcycle": (136, 255, 0),
    "Bus":        (0,   149, 255),
    "Truck":      (255, 95,  191),
}

# ═══════════════════════════════════════════════════════════
#  THÈME DARK
# ═══════════════════════════════════════════════════════════
BG        = "#070a0f"
SURFACE   = "#0d1117"
BORDER    = "#1c2333"
TEXT      = "#e8edf5"
MUTED     = "#4a5568"
ACCENT    = "#00d4ff"


# ═══════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════════
class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Vehicle Counter — YOLO26s")
        self.root.configure(bg=BG)
        self.root.geometry("1300x820")
        self.root.minsize(1100, 700)

        # ── État ──
        self.running      = False
        self.model        = None
        self.cap          = None
        self.counts       = defaultdict(int)
        self.tracked_ids  = {}
        self.crossed_ids  = set()
        self.fps_history  = deque(maxlen=30)
        self.count_history = {k: deque(maxlen=60) for k in VEHICLE_CLASSES.values()}
        self.log_entries  = []
        self.prev_time    = time.time()
        self.current_fps  = 0.0

        # ── Build UI ──
        self._build_ui()
        self._load_model()

    # ───────────────────────────────────────────────────────
    #  BUILD UI
    # ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── HEADER ──────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG, pady=14)
        hdr.pack(fill="x", padx=24)

        tk.Label(hdr, text="🚗  VEHICLE COUNTER", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text="  YOLO26s · CPU Mode · COCO Dataset",
                 font=("Helvetica", 10), bg=BG, fg=MUTED).pack(side="left", pady=6)

        self.status_label = tk.Label(hdr, text="⚪  ARRÊTÉ",
                                     font=("Courier", 10, "bold"), bg=BG, fg=MUTED)
        self.status_label.pack(side="right")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── BODY ────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Colonne gauche (vidéo + log)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # Colonne droite (métriques + graphes)
        right = tk.Frame(body, bg=BG, width=380)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        self._build_video_panel(left)
        self._build_log_panel(left)
        self._build_metrics_panel(right)
        self._build_graph_panel(right)
        self._build_controls(right)

    # ── VIDÉO ────────────────────────────────────────────
    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=SURFACE, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(frame, text="FLUX VIDÉO", font=("Courier", 8),
                 bg=SURFACE, fg=MUTED).pack(anchor="w", padx=12, pady=(8, 0))

        self.video_label = tk.Label(frame, bg="#000000", text="📷\nEn attente de la caméra...",
                                    font=("Helvetica", 12), fg=MUTED, compound="center")
        self.video_label.pack(fill="both", expand=True, padx=8, pady=(4, 8))

    # ── LOG ──────────────────────────────────────────────
    def _build_log_panel(self, parent):
        frame = tk.Frame(parent, bg=SURFACE,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x")

        header = tk.Frame(frame, bg=SURFACE)
        header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(header, text="JOURNAL D'ACTIVITÉ", font=("Courier", 8),
                 bg=SURFACE, fg=MUTED).pack(side="left")
        tk.Button(header, text="EFFACER", font=("Courier", 7), bg=BORDER,
                  fg=MUTED, bd=0, padx=8, pady=2, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(frame, height=5, bg=SURFACE, fg=TEXT,
                                font=("Courier", 9), bd=0, padx=8,
                                state="disabled", wrap="word",
                                insertbackground=TEXT)
        self.log_text.pack(fill="x", padx=4, pady=(0, 8))

        # Tags couleur
        for name, col in COLORS_HEX.items():
            self.log_text.tag_config(name, foreground=col)
        self.log_text.tag_config("time", foreground=MUTED)
        self.log_text.tag_config("ok",   foreground="#00ff88")

    # ── MÉTRIQUES ────────────────────────────────────────
    def _build_metrics_panel(self, parent):
        frame = tk.Frame(parent, bg=SURFACE,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(frame, text="COMPTEURS", font=("Courier", 8),
                 bg=SURFACE, fg=MUTED).pack(anchor="w", padx=12, pady=(8, 6))

        self.count_labels = {}
        self.bar_vars     = {}
        self.bar_fills    = {}

        for vtype in ["Car", "Motorcycle", "Bus", "Truck"]:
            col = COLORS_HEX[vtype]
            row = tk.Frame(frame, bg=SURFACE)
            row.pack(fill="x", padx=12, pady=3)

            # Icône + Nom
            tk.Label(row, text=f"{ICONS[vtype]} {vtype:<12}",
                     font=("Helvetica", 11, "bold"), bg=SURFACE,
                     fg=col, width=16, anchor="w").pack(side="left")

            # Compteur
            lbl = tk.Label(row, text="0", font=("Courier", 18, "bold"),
                           bg=SURFACE, fg=col, width=4, anchor="e")
            lbl.pack(side="right")
            self.count_labels[vtype] = lbl

            # Barre de progression
            bar_frame = tk.Frame(frame, bg=BORDER, height=5)
            bar_frame.pack(fill="x", padx=12, pady=(0, 3))
            bar_frame.pack_propagate(False)

            fill = tk.Frame(bar_frame, bg=col, height=5)
            fill.place(x=0, y=0, relheight=1, relwidth=0)
            self.bar_fills[vtype] = fill

        # TOTAL
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=12, pady=6)
        total_row = tk.Frame(frame, bg=SURFACE)
        total_row.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(total_row, text="TOTAL", font=("Courier", 10, "bold"),
                 bg=SURFACE, fg=TEXT).pack(side="left")
        self.total_label = tk.Label(total_row, text="0",
                                    font=("Courier", 22, "bold"),
                                    bg=SURFACE, fg=TEXT)
        self.total_label.pack(side="right")

    # ── GRAPHE FPS ───────────────────────────────────────
    def _build_graph_panel(self, parent):
        frame = tk.Frame(parent, bg=SURFACE,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", pady=(0, 10))

        header = tk.Frame(frame, bg=SURFACE)
        header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(header, text="PERFORMANCE FPS", font=("Courier", 8),
                 bg=SURFACE, fg=MUTED).pack(side="left")
        self.fps_label = tk.Label(header, text="— FPS",
                                  font=("Courier", 9, "bold"),
                                  bg=SURFACE, fg="#00ff88")
        self.fps_label.pack(side="right")

        self.fps_canvas = tk.Canvas(frame, bg=SURFACE, height=80,
                                    highlightthickness=0, bd=0)
        self.fps_canvas.pack(fill="x", padx=12, pady=(0, 8))

    # ── CONTRÔLES ────────────────────────────────────────
    def _build_controls(self, parent):
        frame = tk.Frame(parent, bg=SURFACE,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x")

        tk.Label(frame, text="CONTRÔLES", font=("Courier", 8),
                 bg=SURFACE, fg=MUTED).pack(anchor="w", padx=12, pady=(8, 6))

        btn_row = tk.Frame(frame, bg=SURFACE)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        self.start_btn = tk.Button(
            btn_row, text="▶  DÉMARRER",
            font=("Helvetica", 10, "bold"),
            bg=ACCENT, fg="#000000",
            bd=0, padx=16, pady=8,
            cursor="hand2", activebackground="#00aacc",
            command=self._start
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_row, text="⏹  ARRÊTER",
            font=("Helvetica", 10, "bold"),
            bg=BORDER, fg=TEXT,
            bd=0, padx=16, pady=8,
            cursor="hand2", state="disabled",
            command=self._stop
        )
        self.stop_btn.pack(side="left", padx=(0, 8))

        self.reset_btn = tk.Button(
            btn_row, text="↺",
            font=("Helvetica", 12, "bold"),
            bg=BORDER, fg="#ff3b5c",
            bd=0, padx=12, pady=8,
            cursor="hand2",
            command=self._reset
        )
        self.reset_btn.pack(side="left")

        # Source info
        src = f"Webcam #{SOURCE}" if isinstance(SOURCE, int) else SOURCE
        tk.Label(frame, text=f"Source : {src}  |  Conf : {CONF_THRESH}  |  Taille : {IMGSZ}px",
                 font=("Courier", 7), bg=SURFACE, fg=MUTED).pack(pady=(0, 8))

    # ───────────────────────────────────────────────────────
    #  CHARGEMENT DU MODÈLE
    # ───────────────────────────────────────────────────────
    def _load_model(self):
        self._log("Chargement de YOLO26s...", "time")

        def load():
            try:
                self.model = YOLO("yolo26s.pt")
                self.root.after(0, lambda: self._log("✅ Modèle YOLO26s chargé !", "ok"))
                self.root.after(0, lambda: self.status_label.config(
                    text="✅  PRÊT", fg="#00ff88"))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"❌ Erreur : {e}", "time"))

        threading.Thread(target=load, daemon=True).start()

    # ───────────────────────────────────────────────────────
    #  DÉMARRAGE / ARRÊT
    # ───────────────────────────────────────────────────────
    def _start(self):
        if self.model is None:
            self._log("⚠ Modèle pas encore chargé, patiente...", "time")
            return

        self.cap = cv2.VideoCapture(SOURCE)
        if not self.cap.isOpened():
            self._log("❌ Impossible d'ouvrir la caméra !", "time")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="🔴  EN COURS", fg="#ff3b5c")
        self._log("▶ Détection démarrée — appuie sur ARRÊTER pour stopper.", "ok")

        threading.Thread(target=self._detect_loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="⚪  ARRÊTÉ", fg=MUTED)
        self._log("⏹ Détection arrêtée.", "time")
        self._print_summary()

    def _reset(self):
        self._stop()
        self.counts.clear()
        self.tracked_ids.clear()
        self.crossed_ids.clear()
        self.fps_history.clear()
        for q in self.count_history.values():
            q.clear()
        self._clear_log()
        self._update_metrics()
        self._draw_fps_graph()

    # ───────────────────────────────────────────────────────
    #  BOUCLE DE DÉTECTION
    # ───────────────────────────────────────────────────────
    def _detect_loop(self):
        self.prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            line_y = int(h * LINE_RATIO)

            # ── Inférence YOLO ──
            results = self.model.track(
                frame,
                persist=True,
                conf=CONF_THRESH,
                iou=0.45,
                imgsz=IMGSZ,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False
            )

            if (results[0].boxes is not None
                    and results[0].boxes.id is not None):
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                for box, tid, cls_id, conf in zip(boxes, ids, classes, confs):
                    label = VEHICLE_CLASSES.get(cls_id)
                    if not label:
                        continue

                    color = COLORS_BGR[label]
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1+x2)//2, int(y2)

                    # Franchissement ligne
                    side = "above" if cy < line_y else "below"
                    if tid not in self.tracked_ids:
                        self.tracked_ids[tid] = side
                    else:
                        if (self.tracked_ids[tid] != side
                                and tid not in self.crossed_ids):
                            self.counts[label] += 1
                            self.crossed_ids.add(tid)
                            msg = f"{ICONS[label]} {label} #{tid} → Total : {self.counts[label]}"
                            self.root.after(0, lambda m=msg, l=label: self._log(m, l))
                        self.tracked_ids[tid] = side

                    # Dessin bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    tag = f"{label} #{tid}  {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                    cv2.putText(frame, tag, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

            # Ligne de comptage
            cv2.line(frame, (0, line_y), (w, line_y), (0, 60, 255), 2)
            cv2.putText(frame, "COUNTING LINE", (w//2-80, line_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now
            self.current_fps = fps
            self.fps_history.append(fps)

            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,136), 2, cv2.LINE_AA)

            # Mettre à jour l'UI dans le thread principal
            self.root.after(0, lambda f=frame.copy(): self._update_frame(f))
            self.root.after(0, self._update_metrics)
            self.root.after(0, self._draw_fps_graph)

        self.root.after(0, lambda: self.video_label.config(
            text="📷\nCaméra arrêtée", image="", compound="center"))

    # ───────────────────────────────────────────────────────
    #  MISE À JOUR AFFICHAGE
    # ───────────────────────────────────────────────────────
    def _update_frame(self, frame):
        """Affiche le frame OpenCV dans le Label Tkinter."""
        lbl_w = self.video_label.winfo_width()
        lbl_h = self.video_label.winfo_height()
        if lbl_w < 2 or lbl_h < 2:
            return

        h, w = frame.shape[:2]
        scale = min(lbl_w/w, lbl_h/h)
        nw, nh = int(w*scale), int(h*scale)
        frame = cv2.resize(frame, (nw, nh))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.video_label.config(image=img, text="")
        self.video_label.image = img

    def _update_metrics(self):
        total = sum(self.counts.values())
        self.total_label.config(text=str(total))

        for vtype in ["Car", "Motorcycle", "Bus", "Truck"]:
            v = self.counts[vtype]
            self.count_labels[vtype].config(text=str(v))

            # Barre
            pct = (v / total) if total > 0 else 0
            bar = self.bar_fills[vtype]
            bar.place(relwidth=pct)

            # Historique
            self.count_history[vtype].append(v)

        fps = self.current_fps
        col = "#00ff88" if fps >= 15 else "#ff9500" if fps >= 8 else "#ff3b5c"
        self.fps_label.config(
            text=f"{fps:.1f} FPS" if fps > 0 else "— FPS", fg=col)

    def _draw_fps_graph(self):
        c = self.fps_canvas
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height() or 80
        if W < 2:
            return

        data = list(self.fps_history)
        if len(data) < 2:
            return

        max_fps = max(max(data), 30)
        n = len(data)

        # Zones de couleur de fond
        zones = [(30, "#001a0a"), (15, "#1a0f00"), (0, "#1a0008")]
        for threshold, col in zones:
            y = H - int((threshold/max_fps)*H)
            c.create_rectangle(0, y, W, H, fill=col, outline="")

        # Grilles
        for thresh in [8, 15, 24]:
            y = H - int((thresh/max_fps)*H)
            c.create_line(0, y, W, y, fill=BORDER, dash=(3,4))
            color = "#00ff88" if thresh==24 else "#ff9500" if thresh==15 else "#ff3b5c"
            c.create_text(W-28, y-6, text=f"{thresh}", fill=color,
                          font=("Courier", 7))

        # Courbe FPS
        pts = []
        for i, v in enumerate(data):
            x = int(i/(n-1)*W)
            y = H - int((v/max_fps)*H*0.9) - 2
            pts.extend([x, y])

        if len(pts) >= 4:
            # Remplissage
            fill_pts = [0, H] + pts + [W, H]
            c.create_polygon(fill_pts, fill="#003322", outline="")
            # Ligne
            c.create_line(pts, fill="#00ff88", width=2, smooth=True)

        # Valeur actuelle
        if data:
            last = data[-1]
            col = "#00ff88" if last >= 15 else "#ff9500" if last >= 8 else "#ff3b5c"
            c.create_text(W//2, H//2, text=f"{last:.0f} FPS",
                          fill=col, font=("Courier", 11, "bold"))

    # ───────────────────────────────────────────────────────
    #  LOG
    # ───────────────────────────────────────────────────────
    def _log(self, msg, tag=""):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("1.0", f"[{now}]  {msg}\n", tag)
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _print_summary(self):
        total = sum(self.counts.values())
        lines = [
            "─" * 36,
            "  📊 RÉSUMÉ FINAL",
            "─" * 36,
        ]
        for vtype in ["Car", "Motorcycle", "Bus", "Truck"]:
            lines.append(f"  {ICONS[vtype]} {vtype:<14} : {self.counts[vtype]}")
        lines += [f"  {'TOTAL':<16} : {total}", "─" * 36]
        for line in lines:
            self._log(line, "time")


# ═══════════════════════════════════════════════════════════
#  PANNEAU D'EXPLICATION : QU'EST-CE QUE LE COMMAND LINE ?
# ═══════════════════════════════════════════════════════════
class CommandLineExplainer:
    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.title("🖥️ C'est quoi le Command Line ?")
        self.win.configure(bg=BG)
        self.win.geometry("700x600")
        self._build()

    def _build(self):
        # Scrollable frame
        canvas = tk.Canvas(self.win, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(self.win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        frame = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        p = 24  # padding

        # Titre
        tk.Label(frame, text="🖥️  C'est quoi le Command Line ?",
                 font=("Helvetica", 16, "bold"), bg=BG, fg=ACCENT).pack(
                     anchor="w", padx=p, pady=(20, 4))

        tk.Label(frame,
                 text="Le Command Line (Terminal / Invite de commandes / Console)\n"
                      "c'est une fenêtre noire où tu tapes du texte pour contrôler\n"
                      "ton PC — sans cliquer. C'est INDISPENSABLE pour lancer Python.",
                 font=("Helvetica", 10), bg=BG, fg="#8899aa",
                 justify="left").pack(anchor="w", padx=p, pady=(0, 16))

        steps = [
            ("1", "Ouvrir le Terminal",
             "Windows : touche Windows → tape 'cmd' → Entrée\n"
             "Mac     : Cmd+Espace → tape 'Terminal' → Entrée\n"
             "Linux   : Ctrl+Alt+T",
             None),

            ("2", "Installer les librairies (une seule fois)",
             "Tape ces commandes dans le terminal :",
             "pip install ultralytics\npip install opencv-python\npip install pillow"),

            ("3", "Aller dans ton dossier (cd = change directory)",
             "Tu dois te placer là où est ton fichier .py :",
             "cd C:\\Users\\TonNom\\Desktop\\MonProjet\n\n"
             "# Astuce Windows : Shift + Clic droit dans ton dossier\n"
             "# → 'Ouvrir PowerShell ici' → tu es déjà au bon endroit !"),

            ("4", "Lancer ton script Python",
             "Une fois dans le bon dossier :",
             "python vehicle_counter_gui.py"),

            ("5", "Commandes utiles à retenir",
             None,
             "dir              # Voir les fichiers (Windows)\n"
             "ls               # Voir les fichiers (Mac/Linux)\n"
             "cd ..            # Remonter d'un dossier\n"
             "python --version # Voir ta version Python\n"
             "Ctrl+C           # Arrêter un script en cours"),
        ]

        for num, title, desc, code in steps:
            row = tk.Frame(frame, bg=BG)
            row.pack(fill="x", padx=p, pady=6)

            badge = tk.Label(row, text=num, font=("Helvetica", 11, "bold"),
                             bg=ACCENT, fg="#000", width=2, pady=2)
            badge.pack(side="left", anchor="n", pady=2)

            content = tk.Frame(row, bg=BG)
            content.pack(side="left", fill="x", expand=True, padx=(12, 0))

            tk.Label(content, text=title, font=("Helvetica", 11, "bold"),
                     bg=BG, fg=TEXT, anchor="w").pack(fill="x")

            if desc:
                tk.Label(content, text=desc, font=("Helvetica", 9),
                         bg=BG, fg="#7a8fa8", justify="left",
                         anchor="w").pack(fill="x", pady=2)

            if code:
                term = tk.Frame(content, bg="#050810",
                                highlightbackground="#1c2a3a",
                                highlightthickness=1)
                term.pack(fill="x", pady=4)
                tk.Label(term, text=code,
                         font=("Courier", 9), bg="#050810",
                         fg="#00d4ff", justify="left",
                         padx=12, pady=10).pack(anchor="w")

        # Tip final
        tip = tk.Frame(frame, bg="#0a1a22",
                       highlightbackground="#0a3a4a",
                       highlightthickness=1)
        tip.pack(fill="x", padx=p, pady=(8, 24))
        tk.Label(tip,
                 text="💡  Astuce pro : utilise VS Code. Tu as un terminal intégré\n"
                      "    directement dans l'éditeur (menu Terminal → Nouveau terminal).",
                 font=("Helvetica", 9), bg="#0a1a22", fg="#8abecc",
                 justify="left", padx=16, pady=12).pack(anchor="w")


# ═══════════════════════════════════════════════════════════
#  FENÊTRE D'ACCUEIL
# ═══════════════════════════════════════════════════════════
class SplashScreen:
    def __init__(self, root, on_start):
        self.win = tk.Toplevel(root)
        self.win.title("Vehicle Counter — Bienvenue")
        self.win.configure(bg=BG)
        self.win.geometry("500x400")
        self.win.resizable(False, False)
        self._build(on_start)

    def _build(self, on_start):
        tk.Label(self.win, text="🚗", font=("Helvetica", 48),
                 bg=BG).pack(pady=(40, 0))

        tk.Label(self.win, text="VEHICLE COUNTER",
                 font=("Helvetica", 20, "bold"),
                 bg=BG, fg=ACCENT).pack()

        tk.Label(self.win, text="YOLO26s · Python · Interface Graphique",
                 font=("Courier", 9), bg=BG, fg=MUTED).pack(pady=4)

        tk.Frame(self.win, bg=BORDER, height=1).pack(fill="x", padx=40, pady=20)

        tk.Label(self.win,
                 text="Cette application détecte et compte\n"
                      "les voitures, motos, bus et camions\n"
                      "en temps réel via ta webcam.",
                 font=("Helvetica", 10), bg=BG, fg="#8899aa").pack()

        tk.Frame(self.win, bg=BG).pack(expand=True)

        btn_frame = tk.Frame(self.win, bg=BG)
        btn_frame.pack(pady=30)

        tk.Button(
            btn_frame, text="▶  LANCER L'APPLICATION",
            font=("Helvetica", 11, "bold"),
            bg=ACCENT, fg="#000", bd=0,
            padx=24, pady=10, cursor="hand2",
            command=lambda: [self.win.destroy(), on_start()]
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="🖥️  Command Line ?",
            font=("Helvetica", 10),
            bg=BORDER, fg=TEXT, bd=0,
            padx=16, pady=10, cursor="hand2",
            command=lambda: CommandLineExplainer(self.win)
        ).pack(side="left", padx=8)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.withdraw()   # Cache la fenêtre principale pendant le splash

    app = None

    def launch():
        root.deiconify()
        nonlocal app
        app = VehicleCounterApp(root)

        # Bouton "Command Line ?" dans la barre de menu
        menu = tk.Menu(root)
        root.config(menu=menu)
        help_menu = tk.Menu(menu, tearoff=0, bg=SURFACE, fg=TEXT)
        menu.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(
            label="🖥️  C'est quoi le Command Line ?",
            command=lambda: CommandLineExplainer(root)
        )
        help_menu.add_separator()
        help_menu.add_command(label="Quitter", command=root.quit)

    SplashScreen(root, launch)
    root.mainloop()


if __name__ == "__main__":
    main()