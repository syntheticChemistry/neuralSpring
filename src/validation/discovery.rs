// SPDX-License-Identifier: AGPL-3.0-or-later

//! Primal Unix socket discovery (biomeOS 5-tier order).

use std::path::{Path, PathBuf};

use crate::config;

/// Result of attempting to discover a primal's Unix socket.
#[derive(Debug, Clone)]
pub enum DiscoveryResult {
    /// Socket found at the given path.
    Found(PathBuf),
    /// Primal not running — no socket found.
    NotFound {
        /// Primal name used in the search.
        primal: String,
        /// Directories that were probed.
        searched: Vec<PathBuf>,
    },
}

impl DiscoveryResult {
    /// Returns `true` if the primal was discovered.
    #[must_use]
    pub const fn is_found(&self) -> bool {
        matches!(self, Self::Found(_))
    }
}

/// Discover a primal's Unix socket using the biomeOS 5-tier discovery order.
///
/// 1. `$BIOMEOS_ORCHESTRATOR_SOCKET` env override
/// 2. `$XDG_RUNTIME_DIR/biomeos/{primal}*.sock`
/// 3. `{temp_dir}/biomeos/{primal}*.sock`
/// 4. `$XDG_RUNTIME_DIR/{primal}/*.sock` (legacy)
/// 5. `{temp_dir}/{primal}-*.sock` (legacy)
///
/// Socket matching uses both the niche name (e.g. `neuralspring`) and the
/// hyphenated `CARGO_PKG_NAME` form (e.g. `neural-spring`) to handle springs
/// whose binary name differs from their niche name.
#[must_use]
pub fn discover_primal_socket(primal: &str) -> DiscoveryResult {
    let mut searched = Vec::new();

    // Tier 0: explicit orchestrator socket override
    if let Ok(override_path) = std::env::var(config::ENV_BIOMEOS_ORCHESTRATOR) {
        let p = PathBuf::from(&override_path);
        searched.push(p.clone());
        if p.exists() {
            return DiscoveryResult::Found(p);
        }
    }

    let alt_name = primal_to_pkg_name(primal);

    if let Ok(xdg) = std::env::var(config::ENV_XDG_RUNTIME_DIR) {
        let biomeos_dir = PathBuf::from(&xdg).join(config::BIOMEOS_SOCKET_SUBDIR);
        searched.push(biomeos_dir.clone());
        if let Some(sock) = find_socket_in_dir(&biomeos_dir, primal, alt_name.as_deref()) {
            return DiscoveryResult::Found(sock);
        }

        let legacy_dir = PathBuf::from(&xdg).join(primal);
        searched.push(legacy_dir.clone());
        if let Some(sock) = find_socket_in_dir(&legacy_dir, primal, alt_name.as_deref()) {
            return DiscoveryResult::Found(sock);
        }
    }

    let tmp_biomeos = std::env::temp_dir().join(config::BIOMEOS_SOCKET_SUBDIR);
    searched.push(tmp_biomeos.clone());
    if let Some(sock) = find_socket_in_dir(&tmp_biomeos, primal, alt_name.as_deref()) {
        return DiscoveryResult::Found(sock);
    }

    let tmp_legacy = std::env::temp_dir();
    searched.push(tmp_legacy.clone());
    if let Some(sock) = find_socket_in_dir(&tmp_legacy, primal, alt_name.as_deref()) {
        return DiscoveryResult::Found(sock);
    }

    DiscoveryResult::NotFound {
        primal: primal.to_string(),
        searched,
    }
}

/// Convert a niche name (e.g. `neuralspring`) to its probable `CARGO_PKG_NAME`
/// form (e.g. `neural-spring`). Returns `None` if the name contains no
/// recognizable camelCase boundary (i.e. it's already a simple name like
/// `beardog` that doesn't need an alternate form).
fn primal_to_pkg_name(niche: &str) -> Option<String> {
    let known = [
        ("neuralspring", "neural-spring"),
        ("hotspring", "hot-spring"),
        ("wetspring", "wet-spring"),
        ("groundspring", "ground-spring"),
        ("airspring", "air-spring"),
        ("healthspring", "health-spring"),
        ("ludospring", "ludo-spring"),
        ("primalspring", "primal-spring"),
        ("esotericwebb", "esoteric-webb"),
    ];
    for &(niche_name, pkg_name) in &known {
        if niche == niche_name {
            return Some(pkg_name.to_string());
        }
    }
    None
}

fn find_socket_in_dir(dir: &Path, primal: &str, alt_name: Option<&str>) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.ends_with(".sock") {
            continue;
        }
        if name_str.contains(primal) {
            return Some(entry.path());
        }
        if let Some(alt) = alt_name {
            if name_str.contains(alt) {
                return Some(entry.path());
            }
        }
    }
    None
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn discovery_result_is_found_when_socket_exists() {
        let dir = std::env::temp_dir().join(format!(
            "ns_disc_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let sock = dir.join("fake.sock");
        std::fs::write(&sock, b"").expect("touch socket file");

        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(sock.to_string_lossy().as_ref()),
            || {
                let result = discover_primal_socket("anything");
                assert!(result.is_found());
                if let DiscoveryResult::Found(path) = result {
                    assert_eq!(path, sock);
                }
            },
        );

        std::fs::remove_file(&sock).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[serial_test::serial]
    fn discover_finds_socket_in_xdg_biomeos_dir() {
        let dir = std::env::temp_dir().join(format!(
            "ns_xdg_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        let biomeos = dir.join(crate::config::BIOMEOS_SOCKET_SUBDIR);
        std::fs::create_dir_all(&biomeos).expect("mkdir biomeos");
        let sock = biomeos.join("neuralspring-test.sock");
        std::fs::write(&sock, b"").expect("touch socket");

        temp_env::with_var(
            crate::config::ENV_XDG_RUNTIME_DIR,
            Some(dir.to_string_lossy().as_ref()),
            || {
                let result = discover_primal_socket("neuralspring-test");
                assert!(result.is_found());
                if let DiscoveryResult::Found(path) = result {
                    assert_eq!(path, sock);
                }
            },
        );

        std::fs::remove_file(&sock).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discovery_not_found_for_fake_primal() {
        let result = discover_primal_socket("nonexistent_primal_xyz");
        assert!(!result.is_found());
    }

    #[test]
    fn discovery_not_found_lists_searched_paths() {
        let result = discover_primal_socket("nonexistent_primal_xyz");
        assert!(!result.is_found());
        if let DiscoveryResult::NotFound { primal, searched } = result {
            assert_eq!(primal, "nonexistent_primal_xyz");
            assert!(!searched.is_empty());
        }
    }

    #[test]
    fn primal_to_pkg_name_known() {
        assert_eq!(
            primal_to_pkg_name("neuralspring"),
            Some("neural-spring".to_string())
        );
        assert_eq!(
            primal_to_pkg_name("hotspring"),
            Some("hot-spring".to_string())
        );
        assert_eq!(primal_to_pkg_name("beardog"), None);
    }

    #[test]
    fn primal_to_pkg_name_unknown() {
        assert_eq!(primal_to_pkg_name("unknownprimal"), None);
    }

    #[test]
    fn discovery_result_is_found_false_for_not_found() {
        let result = discover_primal_socket("nonexistent_primal_xyz");
        assert!(!result.is_found());
    }

    #[test]
    #[serial_test::serial]
    fn discover_finds_socket_in_temp_biomeos_dir() {
        let unique = format!(
            "ns_tmp_bio_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        );
        let biomeos = std::env::temp_dir().join(crate::config::BIOMEOS_SOCKET_SUBDIR);
        std::fs::create_dir_all(&biomeos).expect("mkdir biomeos");
        let sock = biomeos.join(format!("{unique}.sock"));
        std::fs::write(&sock, b"").expect("touch socket");

        temp_env::with_var(crate::config::ENV_XDG_RUNTIME_DIR, None::<&str>, || {
            let result = discover_primal_socket(&unique);
            assert!(result.is_found());
            if let DiscoveryResult::Found(path) = result {
                assert_eq!(path, sock);
            }
        });

        std::fs::remove_file(&sock).ok();
    }

    #[test]
    #[serial_test::serial]
    fn discover_finds_alt_pkg_name_socket() {
        let dir = std::env::temp_dir().join(format!(
            "ns_alt_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        let biomeos = dir.join(crate::config::BIOMEOS_SOCKET_SUBDIR);
        std::fs::create_dir_all(&biomeos).expect("mkdir");
        let sock = biomeos.join("neural-spring-probe.sock");
        std::fs::write(&sock, b"").expect("touch");

        temp_env::with_var(
            crate::config::ENV_XDG_RUNTIME_DIR,
            Some(dir.to_string_lossy().as_ref()),
            || {
                let result = discover_primal_socket("neuralspring");
                assert!(result.is_found());
                if let DiscoveryResult::Found(path) = result {
                    assert_eq!(path, sock);
                }
            },
        );

        std::fs::remove_file(&sock).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[serial_test::serial]
    fn discover_finds_legacy_xdg_primal_dir() {
        let dir = std::env::temp_dir().join(format!(
            "ns_legacy_xdg_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));
        let legacy = dir.join("neuralspring");
        std::fs::create_dir_all(&legacy).expect("mkdir legacy");
        let sock = legacy.join("neuralspring-legacy.sock");
        std::fs::write(&sock, b"").expect("touch");

        temp_env::with_var(
            crate::config::ENV_XDG_RUNTIME_DIR,
            Some(dir.to_string_lossy().as_ref()),
            || {
                let result = discover_primal_socket("neuralspring");
                assert!(result.is_found());
                if let DiscoveryResult::Found(path) = result {
                    assert_eq!(path, sock);
                }
            },
        );

        std::fs::remove_file(&sock).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[serial_test::serial]
    fn discover_orchestrator_override_missing_continues_search() {
        let missing = std::env::temp_dir().join(format!(
            "ns_missing_orch_{}_{:x}.sock",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .subsec_nanos()
        ));

        temp_env::with_var(
            crate::config::ENV_BIOMEOS_ORCHESTRATOR,
            Some(missing.to_string_lossy().as_ref()),
            || {
                let result = discover_primal_socket("nonexistent_primal_xyz");
                assert!(!result.is_found());
            },
        );
    }
}
