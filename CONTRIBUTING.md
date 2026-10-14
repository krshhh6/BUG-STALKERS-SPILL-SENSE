# Contributing to SpillSense

Thank you for your interest in contributing to **SpillSense**! We welcome contributions from developers, geospatial researchers, and maritime technology enthusiasts.

Please take a moment to review this document to streamline the contribution process.

---

## 🧭 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainers.

---

## 🛠️ Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/krshhh6/BUG-STALKERS-SPILL-SENSE.git
cd BUG-STALKERS-SPILL-SENSE
```

### 2. Environment Setup

#### Frontend (React 19 + TypeScript + Vite)
```bash
cd frontend
npm install
npm run dev
```

#### Backend (FastAPI + Python 3.11+)
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## 🌿 Branching Strategy & Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   # or for bug fixes:
   git checkout -b fix/issue-description
   ```
2. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(...)`: A new user-facing or platform feature
   - `fix(...)`: A bug fix
   - `docs(...)`: Documentation changes
   - `chore(...)`: Build process, dependencies, or tool configurations
   - `test(...)`: Adding or updating test suites
3. Keep commits atomic and focused.

---

## 🧪 Testing & Code Quality

Before opening a pull request, ensure the following checks pass:

- **Frontend Build & Lint**:
  ```bash
  cd frontend
  npm run build
  ```
- **Backend Tests**:
  ```bash
  cd backend
  pytest
  ```

---

## 📥 Submitting a Pull Request (PR)

1. Push your branch to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
2. Open a Pull Request against the `main` branch of this repository.
3. Fill out the PR template completely, referencing any related issue numbers (`Closes #123`).
4. Ensure CI checks pass.
5. Address code review feedback promptly.

---

## 💡 Reporting Issues

- **Bug Reports**: Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) with steps to reproduce, expected behavior, and environment details.
- **Feature Requests**: Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md) describing the user value and technical proposal.

Thank you for helping keep sovereign maritime waters safe! 🌊
