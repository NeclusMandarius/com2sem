#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 14.02.2021

import csv
import os

from .conf import Configuration

from tkinter import *
from tkinter import filedialog
from tkinter import ttk, messagebox


class TrainingDataCreator:
    configuration: Configuration = None
    inverse_mapping = None
    training_data = None

    filtered_entries = False

    def __init__(self):
        self.tk = Tk("TrainingDataCreator")
        self.tk.title("TrainingDataCreator")
        self.tk.wm_iconify()

        file = filedialog.askopenfilename()

        self.tk.wm_deiconify()

        self.configuration = Configuration(file)
        try:
            self.configuration.build()
        except:
            messagebox.showerror("ERROR: Invalid Configuration", "Invalid target space configuration given!")
            raise ValueError()

        self.inverse_mapping = dict()

        self.training_data = []
        self.selected_index = -1

        self.menu = Menu(self.tk)
        self.filemenu = Menu(self.menu, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.open)
        self.filemenu.add_command(label="Save As", command=self.save)

        self.menu.add_cascade(label="File", menu=self.filemenu)

        self.tk.config(menu=self.menu)

        self.entries = ttk.Treeview(self.tk, columns=('features',))
        self.entries.heading('features', text='Features')
        self.entries.grid(row=0, columnspan=2)
        vscr = ttk.Scrollbar(self.tk, orient=VERTICAL, command=self.entries.yview)
        self.entries["yscrollcommand"] = vscr.set
        vscr.grid(row=0, column=2, sticky=(S, N))

        self.tree = ttk.Treeview(self.tk)
        self.tree.grid(row=1, column=0, sticky=(N, E, W))

        self.detail_frame = ttk.Frame(self.tk)

        self.word = ttk.Entry(self.detail_frame)
        self.word.grid(row=0, columnspan=2, sticky=EW, padx=5, pady=5)

        self.tk.update()
        self.features_container = Canvas(self.detail_frame, borderwidth=0, height=140, width=200)
        self.features_container.grid(row=1, column=0, columnspan=2, sticky=NSEW)
        self.features = ttk.Frame(self.features_container)
        vscr2 = ttk.Scrollbar(self.detail_frame, orient=VERTICAL, command=self.features_container.yview)
        self.features_container["yscrollcommand"] = vscr2.set
        vscr2.grid(row=1, column=3, sticky=(S, N))
        self.features_container.create_window((0, 0), window=self.features, anchor=NW, tags=("frame",))
        self.feature_vars = {}

        self.button_add = ttk.Button(self.detail_frame, text="Add", command=self.add_entry)
        self.button_change = ttk.Button(self.detail_frame, text="Change", command=self.change_entry)
        self.button_remove = ttk.Button(self.detail_frame, text="Remove", command=self.remove_entry)

        self.button_add.grid(row=3, column=0, sticky=EW)
        self.button_change.grid(row=3, column=1, sticky=EW)
        self.button_remove.grid(row=4, column=0, columnspan=2, sticky=EW)

        self.detail_frame.grid(row=1, column=1, columnspan=2, sticky=NSEW)

        self._build_tree(self.configuration.hierarchy, '', {})

        self._build_entries()

        self.tk.bind("<Control-s>", self.save)
        self.tk.bind("<Control-o>", self.open)

    def add_entry(self):
        word = self.word.get()
        rest = [-1 for _ in range(len(self.configuration.features))]
        for e in self.feature_vars:
            rest[self.configuration.features.index(e)] = self.feature_vars[e].get()
        path = [self.tree.selection()[0]]
        while path[0] != "":
            path.insert(0, self.tree.parent(path[0]))
        for e in path[1:]:
            rest[self.configuration.features.index(e)] = 1
        self.training_data.append([word] + rest + [self.tree.selection()[0]])
        self._build_entries()

    def change_entry(self):
        if self.selected_index < 0:
            return
        word = self.word.get()
        rest = [-1 for _ in range(len(self.configuration.features))]
        for e in self.feature_vars:
            rest[self.configuration.features.index(e)] = self.feature_vars[e].get()
        path = [self.tree.selection()[0]]
        while path[0] != "":
            path.insert(0, self.tree.parent(path[0]))
        for e in path[1:]:
            rest[self.configuration.features.index(e)] = 1
        self.training_data[self.selected_index] = [word] + rest + [self.tree.selection()[0]]
        self._build_entries()

    def remove_entry(self):
        if self.selected_index < 0:
            return
        self.training_data.pop(self.selected_index)
        self._build_entries()

    def update_features(self, i):
        self.feature_vars.clear()
        self.features_container.delete("frame")
        self.features.destroy()
        self.features = ttk.Frame(self.features_container)

        features = self.inverse_mapping[i]
        keys = sorted(features.keys())
        row = 0
        for key in keys:
            label = ttk.Label(self.features, text=key + ":")
            val = "-1" if features[key] is None else str(features[key][0])
            self.feature_vars[key] = StringVar(value=val)
            entry = ttk.Entry(self.features, textvariable=self.feature_vars[key])
            if features[key] is not None and features[key][1]:
                entry["state"] = "disabled"
            label.grid(row=row, column=0)
            entry.grid(row=row, column=1, sticky=EW)
            row += 1
        self.features_container.create_window((0, 0), width=200, window=self.features, anchor=NW, tags=("frame",))
        self.tk.update()
        self.features_container.configure(scrollregion=self.features_container.bbox("all"))

    def _build_tree(self, level, parent, _uses):
        self.inverse_mapping[parent] = _uses.copy()
        if "_uses" in level:
            self.inverse_mapping[parent].update(level["_uses"])
        for i in level:
            if i.startswith("_"):
                continue
            self.tree.insert(parent, 'end', i, text=i, tags=(i,))

            class FeatureUpdater:
                current = i
                this = self

                def fire(self, *_):
                    self.this.update_features(self.current)

                def filter(self, *_):
                    if self.this.filtered_entries:
                        self.this._build_entries()
                    else:
                        self.this._build_entries(filter=self.current)
                    self.this.filtered_entries = not self.this.filtered_entries

            self.tree.tag_bind(i, "<Button-1>", FeatureUpdater().fire)
            self.tree.tag_bind(i, "<Button-2>", FeatureUpdater().filter)
            self.tree.tag_bind(i, "<Button-3>", FeatureUpdater().filter)
            if level[i] is not None:
                self._build_tree(level[i], i, self.inverse_mapping[parent])
            else:
                self.inverse_mapping[i] = self.inverse_mapping[parent]

    def _format_features(self, entry_rest):
        result = []
        ix = 0
        for e in entry_rest:
            if float(e) != -1 and self.configuration.features[ix] not in self.inverse_mapping:
                result.append(f"{self.configuration.features[ix]} = {e}")
            ix += 1
        return ", ".join(result)

    def select_item(self, *_):
        item = self.entries.selection()
        if not item:
            return
        item, ind = item[0].split("+")
        self.selected_index = int(ind)

        selected = self.training_data[self.selected_index]
        if selected[0] != item:
            self.selected_index = -1
            return

        self.tree.selection_set((selected[-1],))
        item = selected[-1]
        while item != "":
            self.tree.item(item, open=True)
            item = self.tree.parent(item)
        self.update_features(selected[-1])

        self.word.delete(0, END)
        self.word.insert(0, selected[0])

        for e in self.feature_vars:
            self.feature_vars[e].set(selected[1 + self.configuration.features.index(e)])

    def _build_entries(self, *_, **kwargs):
        self.entries.delete(*self.entries.get_children())
        ind = 0
        for entry in self.training_data:
            tag = entry[0] + "+" + str(ind)
            if "filter" not in kwargs or entry[1 + self.configuration.features.index(kwargs["filter"])] == "1":
                self.entries.insert('', 'end', tag, text=entry[0],
                                    values=(self._format_features(entry[1:-1]),),
                                    tags=(tag,))
                self.entries.tag_bind(tag, "<ButtonRelease-1>", self.select_item)
            ind += 1

    def save(self, *_):
        filename = filedialog.asksaveasfilename()
        mode = "w" if os.path.exists(filename) else "x"
        with open(filename, mode, encoding="utf-8", newline="") as file:
            wr = csv.writer(file)
            wr.writerow(["word"] + self.configuration.features)
            for i in self.training_data:
                wr.writerow(i[:-1])
        messagebox.showinfo("Save", "Saved successfully!")

    def open(self, *_):
        filename = filedialog.askopenfilename()
        self.training_data.clear()
        with open(filename, encoding="utf-8") as file:
            rd = csv.reader(file)
            index = 0
            for row in rd:
                if row[0].startswith("#") or (index == 0 and row[1:] == self.configuration.features):
                    index += 1
                    continue
                current_state = self.configuration.hierarchy
                current_name = ""
                while current_state:
                    found = False
                    for e in current_state:
                        if e.startswith("_"):
                            continue
                        if float(row[1 + self.configuration.features.index(e)]) == 1:
                            found = True
                            current_state = current_state[e]
                            current_name = e
                            break
                    if not found:
                        break
                index += 1

                if current_name:
                    self.training_data.append(row + [current_name])
                else:
                    print("FAILURE reading line", index, "::", current_state, row)

        self._build_entries()

    def mainloop(self):
        self.tk.mainloop()


if __name__ == "__main__":
    TrainingDataCreator().mainloop()
