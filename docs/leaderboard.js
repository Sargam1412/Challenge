// Load CSV and populate table
async function loadLeaderboard() {

const response = await fetch(
  "https://raw.githubusercontent.com/NoorMajdoub/Challenge/main/leaderboard/leaderboard.csv?ts=" + Date.now()
);

    console.log("response")
    const text = await response.text();
    const rows = text.trim().split("\n");
    const tableBody = document.querySelector("#leaderboard tbody");
    console.log("CSV text:", text);
    tableBody.innerHTML = "";  // clear existing rows

    for (let row of rows) {
        const [team, score] = row.split(/[,|\t]/);
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${team}</td><td>${parseFloat(score).toFixed(4)}</td>`;
        tableBody.appendChild(tr);
    }
}

// Search functionality
document.getElementById("searchBox").addEventListener("input", function() {
    const filter = this.value.toLowerCase();
    const rows = document.querySelectorAll("#leaderboard tbody tr");
    rows.forEach(row => {
        const team = row.cells[0].textContent.toLowerCase();
        row.style.display = team.includes(filter) ? "" : "none";
    });
});

// Simple sort
function sortTable(colIndex) {
    const table = document.getElementById("leaderboard");
    const rows = Array.from(table.tBodies[0].rows);
    const asc = table.getAttribute("data-sort-asc") !== "true";
    rows.sort((a, b) => {
        let x = colIndex === 1 ? parseFloat(a.cells[colIndex].textContent) : a.cells[colIndex].textContent.toLowerCase();
        let y = colIndex === 1 ? parseFloat(b.cells[colIndex].textContent) : b.cells[colIndex].textContent.toLowerCase();
        return asc ? (x > y ? 1 : -1) : (x < y ? 1 : -1);
    });
    rows.forEach(row => table.tBodies[0].appendChild(row));
    table.setAttribute("data-sort-asc", asc);
}

// Load leaderboard on page load
loadLeaderboard();
