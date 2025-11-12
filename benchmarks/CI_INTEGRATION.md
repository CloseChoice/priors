# 🔄 CI/CD Integration - ASV Benchmarks

## 📊 Overview

ASV benchmarks sind jetzt vollständig in die Release-Workflows integriert und laufen automatisch bei jedem Release-Prozess.

## 🚀 Wo die Benchmarks laufen

### 1. **Test Semantic Release** (`feat/**`, `ci/**` branches)
```yaml
Workflow: .github/workflows/test-semantic-release.yml
Trigger: Push zu feat/** oder ci/** branches
Benchmarks: ✅ JA
Deployment: ❌ NEIN (nur als Artifact gespeichert)
Cache: test-results-*
```

### 2. **Production Release** (`main` branch)
```yaml
Workflow: .github/workflows/production-release.yml
Trigger: Push zu main branch
Benchmarks: ✅ JA
Deployment: ✅ JA → GitHub Pages
Cache: prod-results-*
```

## 🔧 Workflow-Struktur

```
commitlint → tests → linting
             ↓
        benchmarks (parallel)
             ↓
     semantic-release
             ↓
    build-python + build-rust
```

### Wichtig:
- **Benchmarks laufen PARALLEL zu semantic-release**
- **Blockieren NICHT den Release-Prozess**
- **Bei Fehler: Release geht weiter**

## 📈 Multi-Branch Support

### Branch-Konfiguration in `asv.conf.json`:
```json
{
  "branches": ["main", "improve-speed", "feat/**", "ci/**"]
}
```

### Im HTML kannst du zwischen Branches wechseln:
1. Öffne https://closechoice.github.io/priors/
2. Klicke auf "Branch" Dropdown
3. Wähle Branch aus (main, improve-speed, etc.)
4. Sieh Benchmark-Ergebnisse für diesen Branch

## 🔄 Wiederverwendbarer Workflow

Die Benchmark-Logik ist jetzt in einem wiederverwendbaren Workflow:

**Datei**: `.github/workflows/benchmarks-reusable.yml`

**Parameter**:
- `deploy_to_pages`: `true/false` - Deploy zu GitHub Pages
- `cache_key_prefix`: `test/prod` - Cache-Präfix für Isolation

**Verwendung**:
```yaml
benchmarks:
  name: Run Benchmarks
  uses: ./.github/workflows/benchmarks-reusable.yml
  with:
    deploy_to_pages: true  # für main branch
    cache_key_prefix: 'prod'
```

## 📦 Caching-Strategie

### Separate Caches für Test und Production:

**Test-Branches**:
```
Key: test-results-feat/my-feature-abc123
Restore: test-results-feat/my-feature-*
         test-results-main-*
         test-results-*
```

**Production (main)**:
```
Key: prod-results-main-abc123
Restore: prod-results-main-*
         prod-results-*
```

**Vorteil**: Test-Benchmarks beeinflussen Production nicht

## 🎯 Deployment-Logik

### Test-Branches (`feat/**`, `ci/**`):
- ✅ Benchmarks laufen
- ✅ Ergebnisse als Artifact gespeichert (90 Tage)
- ❌ KEIN Deployment zu GitHub Pages
- 📊 Im Workflow-Summary sichtbar

### Production (main):
- ✅ Benchmarks laufen
- ✅ Deployment zu GitHub Pages mit `keep_files: true`
- 📈 Alle Branches bleiben im HTML verfügbar
- 🔄 Inkrementelles Update (nur neue Commits)

## 📊 GitHub Pages Setup

### Wichtig: `keep_files: true`

```yaml
- uses: peaceiris/actions-gh-pages@v4
  with:
    keep_files: true  # ← WICHTIG für Multi-Branch!
```

**Ohne `keep_files: true`**:
- Jeder Deploy löscht alte Dateien
- Nur aktueller Branch sichtbar

**Mit `keep_files: true`**:
- Alte Dateien bleiben erhalten
- Alle Branches im HTML sichtbar
- Branch-Wechsel funktioniert

## 🔍 Monitoring

### Im Workflow sehen:
1. Gehe zu **Actions** → **Test/Production Semantic Release**
2. Klicke auf "Run Benchmarks" Job
3. Sieh Logs und Progress

### In Summary sehen:
1. Klicke auf **Summary** Tab
2. Sieh "📊 Benchmark Summary"
3. Anzahl Benchmarks, gecachte Files, etc.

### Artifacts downloaden:
1. Scroll zu "Artifacts" Section
2. Download `asv-results-{branch}-{sha}`
3. Extrahiere und analysiere lokal

## ⚙️ Konfiguration anpassen

### Mehr/Weniger Branches tracken:

**asv.conf.json**:
```json
{
  "branches": [
    "main",           // Production
    "improve-speed",  // Development
    "feat/**"         // Feature branches (wildcard)
  ]
}
```

### Deployment nur für bestimmte Branches:

**production-release.yml**:
```yaml
benchmarks:
  with:
    deploy_to_pages: ${{ github.ref == 'refs/heads/main' }}
```

### Cache-Größe optimieren:

**benchmarks-reusable.yml**:
```yaml
- uses: actions/cache@v4
  with:
    path: .asv/results
    key: ${{ inputs.cache_key_prefix }}-results-${{ github.sha }}
    # Weniger restore-keys = kleinerer Cache
    restore-keys: |
      ${{ inputs.cache_key_prefix }}-results-
```

## 🐛 Troubleshooting

### Problem: Benchmarks schlagen fehl, aber Release geht weiter
**Erwartet**: Benchmarks blockieren nicht den Release
**Lösung**: Check Benchmark-Logs, fixe beim nächsten Push

### Problem: Keine Multi-Branch Ansicht im HTML
**Ursache**: `keep_files: false` oder nur ein Branch deployed
**Lösung**: Setze `keep_files: true` und deploy mehrere Branches

### Problem: Cache zu groß
**Ursache**: Zu viele alte Ergebnisse
**Lösung**: GitHub rotiert automatisch (10GB Limit)

### Problem: Branch nicht im HTML sichtbar
**Ursache**: Branch noch nicht gebenchmarkt oder nicht in Config
**Lösung**:
1. Prüfe `asv.conf.json` → `branches`
2. Pushe zu dem Branch
3. Warte auf Workflow completion

## 📚 Related Files

- [Reusable Workflow](../.github/workflows/benchmarks-reusable.yml)
- [Test Semantic Release](../.github/workflows/test-semantic-release.yml)
- [Production Release](../.github/workflows/production-release.yml)
- [ASV Config](../asv.conf.json)
- [Workflow Guide](./WORKFLOW_GUIDE.md)
- [Local Guide](./LOCAL_GUIDE.md)

## 🎉 Benefits

✅ **Automatisch**: Läuft bei jedem Release
✅ **Inkrementell**: Nur neue Commits benchmarken (schnell!)
✅ **Multi-Branch**: Alle Branches im HTML verfügbar
✅ **Isoliert**: Test und Production getrennt
✅ **Non-Blocking**: Fehler blockieren nicht den Release
✅ **Cached**: Wiederverwendet Ergebnisse
✅ **Transparent**: Summary in jedem Workflow
