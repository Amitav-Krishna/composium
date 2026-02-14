# Agent Instructions

## Planning Modes
- **Plan**: Write `*plan.md` files (default)
- **Execute**: Implement existing plans when user says "execute"
- **Cleanup**: Delete `*plan.md` files when user says "done"

## Key Commands
- Dev: `npm run dev` (in `web/`)
- Tests: `npm run test`
- Kill ports: `.\scripts\kill-ports.ps1`
- Migrations: `.\scripts\migrate.ps1 up`

## Essential Rules
- **Architecture**: Read [system_architecture.md](./system_architecture.md) and [schema_architecture.md](./schema_architecture.md)
- **Styling**: Read [brand.md](./brand.md)
- **API**: Use generated services in `src/api/generated-web-backend`, never raw fetch
- **Schema**: sqlc authority over Prisma (`db/db/models.go`)
- **Media**: Always use presigned URLs, never raw R2 URLs
- **Components**: Use `src/shared/ui/` primitives, one component per file <150 lines
