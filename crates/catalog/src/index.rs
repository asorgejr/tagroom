//! Lookups derived from the catalog's own data, so questions asked once per photo of a big library
//! don't scan a list each time: which stack a photo is in, whether a regular album holds it.
//!
//! They are never saved (the on-disk format is unchanged). Each is built on first use from what
//! the catalog holds, so a catalog read from a snapshot, replayed from the journal or cloned starts
//! without them and builds them when asked. Once built, [`crate::Catalog::apply`] keeps them up to
//! date: the ops that change stacks or album photos (`AddStack`, `RemoveStack`, `SetStack`,
//! `AddAlbum`, `RemoveAlbum`, `SetAlbumPhotos`) update them as they apply, inverses and the
//! rollback of a failed batch included, as those are ops too.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::{AlbumId, Catalog, PhotoId, StackId};

#[derive(Default)]
pub(crate) struct Indexes {
    /// Photo → the stack it is in; `None` when the catalog was read with a photo in two stacks
    /// (only a hand-edited file can hold that: ops refuse it), where a scan says which one counts.
    stacks: OnceLock<Option<HashMap<PhotoId, StackId>>>,
    /// Album → its photos (regular albums; a smart album's list is empty).
    albums: OnceLock<HashMap<AlbumId, HashSet<PhotoId>>>,
    /// The library's keywords for the current revision, rebuilt when `apply` moves it on.
    keywords: Mutex<Option<(u64, Arc<KeywordIndex>)>>,
}

/// A copy builds its own when asked (copies are made to be saved, not queried).
impl Clone for Indexes {
    fn clone(&self) -> Self {
        Indexes::default()
    }
}

/// Derived from the rest of the catalog: two catalogs with the same data are equal whatever was
/// built.
impl PartialEq for Indexes {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for Indexes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stacks = self.stacks.get().map(|m| m.as_ref().map(HashMap::len));
        let keywords = self.keywords.lock().ok().and_then(|k| k.as_ref().map(|(_, i)| i.counts.len()));
        f.debug_struct("Indexes").field("stacks", &stacks).field("albums", &self.albums.get().map(HashMap::len)).field("keywords", &keywords).finish()
    }
}

impl Indexes {
    /// Stack `id` held `old` and holds `new` now (empty: it was added, or is gone).
    pub(crate) fn stack_changed(&mut self, id: StackId, old: &[PhotoId], new: &[PhotoId]) {
        let Some(Some(m)) = self.stacks.get_mut() else { return };
        for p in old {
            if m.get(p) == Some(&id) {
                m.remove(p);
            }
        }
        for p in new {
            m.insert(*p, id);
        }
    }

    /// Album `id` holds `photos` now (`None`: it is gone).
    pub(crate) fn album_changed(&mut self, id: AlbumId, photos: Option<&[PhotoId]>) {
        let Some(m) = self.albums.get_mut() else { return };
        match photos {
            Some(photos) => {
                m.insert(id, photos.iter().copied().collect());
            }
            None => {
                m.remove(&id);
            }
        }
    }
}

/// The library's keywords for one revision: the questions asked on every selection change
/// (suggestions) read the library once instead of scanning every photo each time.
pub(crate) struct KeywordIndex {
    /// Every keyword a photo has, as it was first written, with its application count; sorted by
    /// name, the shape [`crate::Catalog::keywords`] returns.
    counts: Vec<(String, usize)>,
    /// `counts[i]`'s keyword in lower case: what suggestion matching and ordering compare.
    lower: Vec<String>,
    /// Lower-case keyword → its row in `counts`: for spellings that differ only in case,
    /// the one that sorts first.
    ids: HashMap<String, u32>,
    /// Lower-case keyword → the photos that have it, any spelling.
    photos: HashMap<String, Vec<PhotoId>>,
    /// The keyword list's keywords that no photo has (suggestions offer them at count 0).
    listed: Vec<String>,
    /// `listed[i]`'s keyword in lower case.
    listed_lower: Vec<String>,
}

impl KeywordIndex {
    /// The keywords of the library `c` holds in this revision.
    fn of(c: &Catalog) -> Self {
        let mut counts: BTreeMap<String, usize> = Default::default();
        let mut photos: HashMap<String, Vec<PhotoId>> = Default::default();
        for p in c.photos().filter(|p| p.in_library()) {
            for k in &p.meta.keywords {
                *counts.entry(k.clone()).or_default() += 1;
                photos.entry(k.to_lowercase()).or_default().push(p.id);
            }
        }
        let listed: Vec<String> = c.keyword_list.values().filter(|l| !photos.contains_key(&l.path.to_lowercase())).map(|l| l.path.clone()).collect();
        let counts: Vec<(String, usize)> = counts.into_iter().collect();
        let mut lower = Vec::with_capacity(counts.len());
        let mut ids = HashMap::with_capacity(counts.len());
        for (i, (k, _)) in counts.iter().enumerate() {
            let l = k.to_lowercase();
            ids.entry(l.clone()).or_insert(i as u32);
            lower.push(l);
        }
        let listed_lower = listed.iter().map(|k| k.to_lowercase()).collect();
        KeywordIndex { counts, lower, ids, photos, listed, listed_lower }
    }

    /// Every keyword a photo has, as it was first written, with its application count; sorted by name.
    pub(crate) fn counts(&self) -> &[(String, usize)] {
        &self.counts
    }

    /// The photos that have `keyword`, any spelling; empty for one no photo has.
    pub(crate) fn photos_of(&self, keyword: &str) -> &[PhotoId] {
        self.photos.get(&keyword.to_lowercase()).map_or(&[], Vec::as_slice)
    }

    /// The vocabulary suggestions score: every keyword a photo has, then the keyword list's.
    pub(crate) fn rows(&self) -> usize {
        self.counts.len() + self.listed.len()
    }

    /// Row `i`'s keyword in lower case and its count: a photo's keyword while `i < counts.len()`,
    /// a keyword-list entry (count 0) after.
    pub(crate) fn row(&self, i: usize) -> Option<(&str, usize)> {
        if i < self.counts.len() {
            Some((self.lower.get(i).map(String::as_str).unwrap_or(""), self.counts.get(i).map_or(0, |(_, c)| *c)))
        } else {
            self.listed_lower.get(i - self.counts.len()).map(|l| (l.as_str(), 0))
        }
    }

    /// Row `i`'s keyword as first written: the name a suggestion offers.
    pub(crate) fn row_name(&self, i: usize) -> Option<&str> {
        if i < self.counts.len() { self.counts.get(i).map(|(k, _)| k.as_str()) } else { self.listed.get(i - self.counts.len()).map(String::as_str) }
    }

    /// The counts row of a photo's keyword by its lower case: among spellings that differ
    /// only in case, the one that sorts first.
    pub(crate) fn id_of(&self, keyword: &str) -> Option<u32> {
        self.ids.get(keyword).copied()
    }
}

impl Catalog {
    /// The stack photo `photo` is in, by the index (built on first use).
    pub(crate) fn stack_id_of(&self, photo: PhotoId) -> Option<StackId> {
        let m = self.index.stacks.get_or_init(|| {
            let mut m = HashMap::new();
            for s in self.stacks.values() {
                for p in &s.photos {
                    if m.insert(*p, s.id).is_some_and(|other| other != s.id) {
                        return None;
                    }
                }
            }
            Some(m)
        });
        match m {
            Some(m) => m.get(&photo).copied(),
            // in id order, the first stack listing the photo has it
            None => self.stacks.values().find(|s| s.photos.contains(&photo)).map(|s| s.id),
        }
    }

    /// Whether album `album`'s own list holds `photo`, by the index (built on first use).
    pub(crate) fn album_lists(&self, album: AlbumId, photo: PhotoId) -> bool {
        let m = self.index.albums.get_or_init(|| self.albums.values().map(|a| (a.id, a.photos.iter().copied().collect())).collect());
        m.get(&album).is_some_and(|s| s.contains(&photo))
    }

    /// The library's keywords, by the per-revision index (built on first use, rebuilt when an
    /// applied op moves the revision on).
    pub(crate) fn keyword_index(&self) -> Arc<KeywordIndex> {
        let mut held = self.index.keywords.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((rev, index)) = held.as_ref()
            && *rev == self.revision
        {
            return index.clone();
        }
        let index = Arc::new(KeywordIndex::of(self));
        *held = Some((self.revision, index.clone()));
        index
    }
}
