const fs = require("fs")

const res = []
const lbls = {}

for (const fn of process.argv.slice(2)) {
    const fn2 = fn.replace(/\.csv$/, "")
    if (fn2 == fn)
        continue
    const lbl = fn2.replace(/ \(\d+\)/, "").replace(/\d+$/, "")
    const gesture = {
        label: lbl,
        samples: []
    }
    res.push(gesture)
    if (!lbls[lbl])
        lbls[lbl] = 1
    else
        lbls[lbl]++
    for (let line of fs.readFileSync(fn, "utf8").split(/\n/).slice(1)) {
        line = line.trim()
        if (!line)
            continue
        const words = line.split(/,/).map(parseFloat)
        gesture.samples.push(words.slice(1))
    }
}
console.log(lbls)
fs.writeFileSync("data.json", JSON.stringify(res, null, 1))
