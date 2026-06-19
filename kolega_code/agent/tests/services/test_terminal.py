import pytest

from kolega_code.services.terminal import LocalTerminalManager


@pytest.fixture
def manager():
    # connection_manager=None: output broadcasting is skipped in tests.
    return LocalTerminalManager("workspace", "thread", None)


@pytest.mark.asyncio
async def test_exec_command_success(manager):
    result = await manager.exec_command("echo hello world", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "hello world" in result.output
    assert result.session_id is None


@pytest.mark.asyncio
async def test_exec_command_nonzero_exit_code(manager):
    result = await manager.exec_command("exit 7", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code == 7


@pytest.mark.asyncio
async def test_exec_command_failing_command_is_nonzero(manager):
    result = await manager.exec_command("ls /this_path_does_not_exist_xyz", yield_time_ms=5000)
    assert result.status == "exited"
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_long_running_returns_session_then_completes(manager):
    result = await manager.exec_command("echo start; sleep 1; echo done", yield_time_ms=250)
    assert result.status == "running"
    assert result.session_id is not None
    session_id = result.session_id

    for _ in range(40):
        result = await manager.write_stdin(session_id, "", yield_time_ms=2000)
        if result.status == "exited":
            break
    assert result.status == "exited"
    assert result.exit_code == 0
    assert "done" in result.output


@pytest.mark.asyncio
async def test_interactive_stdin(manager):
    result = await manager.exec_command('printf "P> "; read x; echo got=$x', yield_time_ms=400)
    assert result.status == "running"
    assert "P>" in result.output

    result = await manager.write_stdin(result.session_id, "ada\n", yield_time_ms=3000)
    assert result.status == "exited"
    assert "got=ada" in result.output


@pytest.mark.asyncio
async def test_kill_session_interrupt_reports_130(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=300)
    assert result.status == "running"
    killed = await manager.kill_session(result.session_id, "INT")
    assert killed.status == "exited"
    assert killed.exit_code == 130


@pytest.mark.asyncio
async def test_kill_session_term(manager):
    result = await manager.exec_command("sleep 30", yield_time_ms=300)
    killed = await manager.kill_session(result.session_id, "TERM")
    assert killed.status == "exited"
    # SIGTERM -> 143, or SIGKILL fallback -> 137
    assert killed.exit_code in (143, 137)


@pytest.mark.asyncio
async def test_list_sessions_tracks_running_and_clears(manager):
    result = await manager.exec_command("sleep 5", yield_time_ms=200)
    sessions = await manager.list_sessions()
    assert result.session_id in sessions
    assert sessions[result.session_id]["running"] is True

    await manager.kill_session(result.session_id, "TERM")
    assert result.session_id not in await manager.list_sessions()


@pytest.mark.asyncio
async def test_write_stdin_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        await manager.write_stdin("does_not_exist")


@pytest.mark.asyncio
async def test_kill_unknown_session_raises(manager):
    with pytest.raises(KeyError):
        await manager.kill_session("does_not_exist")


@pytest.mark.asyncio
async def test_run_command_convenience_accumulates_output(manager):
    output = await manager.run_command("echo a; echo b; echo c")
    assert "a" in output and "b" in output and "c" in output


@pytest.mark.asyncio
async def test_workdir_is_respected(manager, tmp_path):
    result = await manager.exec_command("pwd", workdir=str(tmp_path), yield_time_ms=3000)
    assert result.status == "exited"
    # macOS resolves symlinks (/var -> /private/var); the leaf dir is enough.
    assert result.output.strip().endswith(tmp_path.name)


@pytest.mark.asyncio
async def test_no_cwd_persistence_between_calls(manager, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    # cd in one call must NOT affect the next (fresh process per exec).
    await manager.exec_command(f"cd {sub}", workdir=str(tmp_path), yield_time_ms=3000)
    result = await manager.exec_command("pwd", workdir=str(tmp_path), yield_time_ms=3000)
    assert result.output.strip().endswith(tmp_path.name)


@pytest.mark.asyncio
async def test_clean_env_overlay(manager):
    result = await manager.exec_command("echo $NO_COLOR-$TERM-$PAGER", yield_time_ms=3000)
    assert "1-dumb-cat" in result.output


@pytest.mark.asyncio
async def test_close_all_terminates_sessions(manager):
    await manager.exec_command("sleep 30", yield_time_ms=200)
    await manager.exec_command("sleep 30", yield_time_ms=200)
    assert len(manager.sessions) == 2
    await manager.close_all()
    assert len(manager.sessions) == 0
