
## display anteriorization based on intermediate saved data
ant = {}
with h5py.File(anteriorization_path, 'r') as h:
    for name in h:
        ant[h] = np.array(h[name])

for name
c,a,p = res.T
keep = np.diff(c, prepend=-1) > 0

c = c[keep]
a = a[keep]
p = p[keep]

fig, axs = pl.subplots(1, 3, figsize=(12,4))
ax = axs[0]
ax.scatter(c, a, c=np.arange(len(c)), s=200)
ax.set_title('anterior')
ax = axs[1]
ax.scatter(c ,p, c=np.arange(len(c)), s=200)
ax.set_title('posterior')
ax = axs[2]
ax.scatter(c, a/p, c=np.arange(len(c)), s=200)
ax.set_title('a/p ratio')
ax.set_ylim([0,4])
ax.set_xlim([-0.3,3.6])
ax.set_xticks(np.arange(0, 3.3, 0.5))
ax.grid(True)
pl.savefig(f'/Users/bdd/Desktop/alpha_{name}.pdf')
